import os
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf

from video_features.models.i3d.extract_i3d import ExtractI3D
from video_features.utils.utils import build_cfg_path


BBOX_EXPAND = 1.1  # Expand by 10%
I3D_EXTRACTOR = None


def crop_to_bbox(frame, bbox, expand=BBOX_EXPAND):
    if bbox is None:
        return frame

    h, w, _ = frame.shape
    bx, by, bw, bh = bbox  # normalized [0,1]

    cx = bx + bw / 2
    cy = by + bh / 2
    bw = bw * expand
    bh = bh * expand

    x1 = max(0.0, cx - bw / 2)
    y1 = max(0.0, cy - bh / 2)
    x2 = min(1.0, cx + bw / 2)
    y2 = min(1.0, cy + bh / 2)

    px1 = int(x1 * w)
    py1 = int(y1 * h)
    px2 = int(x2 * w)
    py2 = int(y2 * h)

    return frame[py1:py2, px1:px2]


def resize_and_pad(frame, target_size=224):
    """
    Resize frame so that the longest side == target_size,
    then pad the shorter side to make a square image.
    """
    h, w, _ = frame.shape

    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_size - new_w
    pad_h = target_size - new_h

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),  # black padding
    )

    assert padded.shape[0] == target_size
    assert padded.shape[1] == target_size

    return padded


def pad_video_to_multiple_of_clipsize(frames, clip_size=32):
    """
    Always pad frames so that:
    - total_frames is divisible by clip_size
    - plus ONE extra clip as safety margin
    """
    num_frames = len(frames)
    if num_frames == 0:
        raise ValueError("No frames to pad.")

    # number of full clips needed (ceil)
    num_clips = (num_frames + clip_size - 1) // clip_size

    # +1 extra clip for I3D internal safety
    target_frames = (num_clips + 1) * clip_size
    pad_len = target_frames - num_frames

    if pad_len > 0:
        frames.extend([frames[-1]] * pad_len)

    assert len(frames) >= clip_size, "Not enough frames after padding."
    assert len(frames) % clip_size == 0, "Total frames not multiple of clip size."

    return frames


def preprocess_video_to_tmp(video_path, tmp_path, clip_size=32, bbox=None):
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Unable to open video: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = crop_to_bbox(frame, bbox)  # optional
        if frame.size == 0:
            continue

        frame = resize_and_pad(frame, target_size=224)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No valid frames in video: {video_path}")

    frames = pad_video_to_multiple_of_clipsize(frames, clip_size)

    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(tmp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (224, 224),
    )

    for f in frames:
        writer.write(f)
    writer.release()

    return tmp_path


def log_warning(warn_file, message):
    with open(warn_file, "a") as f:
        f.write(message + "\n")
        f.flush()


def process_video(
    video_path,
    out_path,
    bbox,
    tmp_dir,
    warn_file,
    num_frames=32,
    streams="rgb+flow",
):
    if out_path.exists():
        return True  # Skip if already processed

    try:
        tmp_video_path = tmp_dir / (video_path.stem + "_proc.mp4")

        tmp_video_path = preprocess_video_to_tmp(
            video_path,
            tmp_video_path,
            clip_size=num_frames,
            bbox=bbox,
        )

        global I3D_EXTRACTOR
        feature_dict = I3D_EXTRACTOR.extract(tmp_video_path)

        if streams == "rgb+flow":
            rgb_feat = feature_dict["rgb"]  # (T, 1024)
            flow_feat = feature_dict["flow"]  # (T, 1024)
            assert rgb_feat.shape[0] > 0 and flow_feat.shape[0] > 0, (
                "No clips extracted."
            )
            output = np.stack([rgb_feat, flow_feat], axis=0)  # (2, T, 1024)
            output = output.mean(axis=1)  # (2, 1024)
        elif streams == "rgb":
            output = feature_dict["rgb"]  # (T, 1024)
            assert output.shape[0] > 0, "No clips extracted."
            output = np.mean(output, axis=0)  # (1024,)
        elif streams == "flow":
            output = feature_dict["flow"]  # (T, 1024)
            assert output.shape[0] > 0, "No clips extracted."
            output = np.mean(output, axis=0)  # (1024,)
        else:
            raise ValueError("No valid streams found in I3D extractor.")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, output.astype(np.float32))

        return True

    except Exception as e:
        if warn_file is not None:
            log_warning(warn_file, f"Error processing {video_path}: {str(e)}")
        print(f"Error processing {video_path}: {str(e)}")
        return False
    finally:
        # Ensure tmp video is removed
        tmp_video_path.unlink(missing_ok=True)


def build_i3d_config(num_frames, device, streams):
    cfg = OmegaConf.load(build_cfg_path("i3d"))
    cfg.stack_size = num_frames
    cfg.step_size = num_frames
    cfg.streams = None if streams == "rgb+flow" else streams
    cfg.flow_type = "raft"
    cfg.device = device
    cfg.extraction_fps = None
    return cfg


def main():
    command_used = " ".join(["python"] + os.sys.argv)

    parser = argparse.ArgumentParser("Extract I3D features with bbox cropping")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--crop-to-bbox", action="store_true", required=True)
    parser.add_argument("--num-frames", type=int, default=32, required=True)
    parser.add_argument("--device", default="cuda:0", required=True)
    parser.add_argument("--streams", choices=["rgb", "flow", "rgb+flow"], required=True)
    parser.add_argument("--overwrite", action="store_true", default=False)

    args = parser.parse_args()

    in_root = Path(args.dataset_root)
    meta = pd.read_csv(in_root / "metadata.csv")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for child in out_root.iterdir():
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)

    tmp_dir = out_root / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    # ---- I3D config ----
    cfg = build_i3d_config(args.num_frames, args.device, args.streams)
    global I3D_EXTRACTOR
    I3D_EXTRACTOR = ExtractI3D(cfg)

    # Warn file
    warn_file = out_root / "warnings.txt"

    for row in tqdm(meta.itertuples(), total=len(meta), desc="Extracting I3D features"):
        video_path = in_root / row.filepath
        out_path = out_root / row.gloss / (Path(row.filepath).stem + ".npy")
        bbox = (row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h)

        process_video(
            video_path,
            out_path,
            bbox if args.crop_to_bbox else None,
            tmp_dir,
            warn_file,
            num_frames=args.num_frames,
            streams=args.streams,
        )

    if args.crop_to_bbox:
        crop_text = (
            "- Videos are cropped using the provided bounding boxes.\n"
            f"- Bounding boxes are expanded by **{(BBOX_EXPAND - 1) * 100:.0f}% on all sides** before cropping."
        )
    else:
        crop_text = "- Videos are not cropped; full frames are used."

    readme_path = out_root / "README.md"
    readme_text = f"""# I3D Video Features

This directory contains pre-extracted I3D spatiotemporal features derived from
the reduced WLASL dataset.

## Feature Description

- **Backbone**: Pretrained I3D
- **Input FPS**: 24 FPS
- **Streams**: {"RGB + Optical Flow" if args.streams == "rgb+flow" else "RGB only" if args.streams == "rgb" else "Optical Flow only"}
- **Clip length**: {args.num_frames} frames
- **Clip stride**: {args.num_frames} frames
- **Pooling strategy**:
  - Global average pooling per clip
  - Average pooling across clips

Each video is represented as a single NumPy array:

- RGB only: `(1024,)`
- RGB + Flow: `(2, 2048)`

## Preprocessing Details

{crop_text}
- Videos are resized to 224x224 pixels.
- Videos are segmented into non-overlapping {args.num_frames}-frame clips.
- If the video length (number of frames) is not a multiple of {args.num_frames}, it is padded by repeating the last frame. One extra clip ({args.num_frames} frames) is always added as a safety margin to every video.


## Directory Structure

```bash
{out_root.as_posix()}/
├── <gloss_label>/
│   ├── <video_id>.npy # shape: (1024,) or (2, 2048)
│   ├── ...
├── warnings.txt # Optional: Warnings during extraction
└── README.md
````

## Command Used

The features were extracted using the following command:

```bash
{command_used}
```
"""

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_text)
    print(f"README saved to: {readme_path.as_posix()}")

    # Delete tmp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
