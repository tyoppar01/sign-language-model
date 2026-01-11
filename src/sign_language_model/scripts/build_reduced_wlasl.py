import argparse
import json
import os
import shutil
from pathlib import Path
from shutil import which
import subprocess

import cv2
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build reduced WLASL dataset with optional FPS normalization"
    )

    parser.add_argument(
        "--wlasl-root",
        type=Path,
        required=True,
        help="Path to original WLASL dataset root",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Path to output reduced dataset",
    )

    parser.add_argument(
        "--glosses",
        type=str,
        required=True,
        help="Comma-separated list of glosses, e.g. 'thin,cool,before,go'",
    )

    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Normalize videos to this FPS (e.g. 24). If omitted, keep original FPS.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it exists",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without copying videos",
    )

    return parser.parse_args()


def parse_glosses(glosses_str):
    """Parse comma-separated glosses string into a list."""
    if glosses_str is None:
        return None
    return [gloss.strip() for gloss in glosses_str.split(",") if gloss.strip()]


def normalize_fps_ffmpeg(src, dst, target_fps):
    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        f"fps={target_fps}",
        "-an",
        str(dst),
    ]


def check_ffpmeg_installed():
    """Check if ffmpeg is installed."""

    if which("ffmpeg") is None:
        raise EnvironmentError("ffmpeg is not installed or not found in PATH.")


def extract_useful_info(sample) -> dict:
    try:
        filepath = sample["filepath"]
        gloss_label = sample["gloss"]["label"]

        frame_rate = sample.get("metadata", {}).get("frame_rate", None)

        detections = sample.get("bounding_box", {}).get("detections", [])
        if len(detections) == 0:
            raise ValueError(f"No bounding box detections found for sample {filepath}")

        bbox_x, bbox_y, bbox_w, bbox_h = detections[0]["bounding_box"]

        return {
            "filepath": filepath,
            "frame_rate": frame_rate,
            "gloss_label": gloss_label,
            "bbox_x": bbox_x,
            "bbox_y": bbox_y,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
        }

    except Exception as e:
        print(
            f"Failed to extract info for sample {sample.get('filepath', 'unknown')}: {e}"
        )
        raise e


def main():
    command_used = " ".join(["python"] + os.sys.argv)
    args = parse_args()

    raw_root = args.wlasl_root
    out_root = args.output_root
    glosses = set(parse_glosses(args.glosses))
    samples_path = raw_root / "samples.json"
    video_out = out_root / "videos"

    if out_root.exists() and args.overwrite and not args.dry_run:
        shutil.rmtree(out_root)

    video_out.mkdir(parents=True, exist_ok=True)

    samples = json.load(open(samples_path))["samples"]
    df_samples = pd.DataFrame([extract_useful_info(s) for s in samples])
    all_glosses = set(df_samples["gloss_label"].unique())

    # Check the provided glosses
    invalid_glosses = glosses - all_glosses
    if invalid_glosses:
        raise ValueError(
            f"The following glosses are not found in the dataset: {invalid_glosses}"
        )

    df_samples = df_samples[df_samples["gloss_label"].isin(glosses)]

    rows = []

    print(f"Selected glosses: {len(glosses)}")
    # Print glosses
    print(f"Glosses: {sorted(glosses)}")
    print(f"Target FPS: {args.target_fps or 'keep original'}")

    # Loop through glosses
    for gloss, df_gloss in df_samples.groupby("gloss_label"):
        out_dir = video_out / gloss
        out_dir.mkdir(parents=True, exist_ok=True)

        for _, row in tqdm(
            df_gloss.iterrows(),
            total=len(df_gloss),
            desc=f"Processing samples for gloss '{gloss}'",
        ):
            src_video = raw_root / row["filepath"]
            if not src_video.exists():
                print(f"Warning: Source video {src_video} does not exist. Skipping.")
                continue

            out_video = out_dir / src_video.name

            if not args.dry_run:
                if args.target_fps is None:
                    shutil.copy(src_video, out_video)
                else:
                    result = subprocess.run(
                        normalize_fps_ffmpeg(
                            src=str(src_video),
                            dst=str(out_video),
                            target_fps=args.target_fps,
                        ),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    if result.returncode != 0:
                        print(f"[FFMPEG ERROR] Failed on {src_video}: {result.stderr}")
                        continue

            cap = cv2.VideoCapture(str(out_video))
            fps = cap.get(cv2.CAP_PROP_FPS)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = round(num_frames / fps if fps > 0 else 0, 2)  # round off

            cap.release()

            rows.append(
                {
                    "filepath": str(out_video.relative_to(out_root).as_posix()),
                    "gloss": gloss,
                    "fps": args.target_fps or fps,
                    "duration": duration,
                    "num_frames": int(num_frames),
                    "video_width": width,
                    "video_height": height,
                    "bbox_x": row["bbox_x"],
                    "bbox_y": row["bbox_y"],
                    "bbox_w": row["bbox_w"],
                    "bbox_h": row["bbox_h"],
                }
            )

    df_out = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_root / "metadata.csv", index=False)

    print(f"\nSaved {len(df_out)} samples to {Path(out_root).as_posix()}")
    print(f"Metadata written to {Path(out_root / 'metadata.csv').as_posix()}")

    # gloss_map.json
    glosses_found = sorted(df_out["gloss"].unique())
    gloss_map = {gloss: idx for idx, gloss in enumerate(glosses_found)}
    with open(out_root / "gloss_map.json", "w") as f:
        json.dump(gloss_map, f, indent=4)
    print(
        f"Gloss map with {len(gloss_map)} entries saved to {Path(out_root / 'gloss_map.json').as_posix()}"
    )

    # readme
    readme_path = out_root / "README.md"

    readme_text = f"""# Reduced WLASL Dataset

This dataset is a task-specific reduced version of the WLASL dataset,
constructed for American Sign Language (ASL) recognition experiments.

## Contents

- `videos/`  
  Video clips organized by gloss label.

- `metadata.csv`  
  Per-sample metadata including:
  - file path
  - gloss label
  - fps (after normalization, if applied)
  - video resolution
  - normalized bounding box coordinates
- `gloss_map.json`  
  Mapping from gloss labels to integer class IDs.

## Dataset Construction

The dataset was generated from the original WLASL dataset using a custom
CLI preprocessing script with the following criteria:

- Glosses restricted to a selected subset
- Samples without bounding boxes were excluded
- Videos optionally normalized to a fixed FPS
- Original videos preserved (no cropping applied)
- Bounding boxes stored as metadata for downstream processing

## Command Used

The dataset was generated using the following command:

```bash
{command_used}
```

## Notes

- Bounding boxes are stored in normalized coordinates (0-1) with top-left origin.
- FPS normalization (if applied) uses frame dropping (no interpolation).

"""

    with open(readme_path, "w") as f:
        f.write(readme_text)

    print(f"README saved to {Path(readme_path).as_posix()}")


if __name__ == "__main__":
    check_ffpmeg_installed()
    main()
