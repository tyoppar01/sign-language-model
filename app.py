import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import json
import math
import tempfile
import av
import threading
import time
import queue
from collections import deque, Counter
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, WebRtcMode


# =========================================================
# 1. MODEL ARCHITECTURE (Exact Copy from Training)
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TemporalAttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return (weights * x).sum(dim=1)


class TransformerEncoderKeypoints(nn.Module):
    def __init__(self, num_classes, T=32, V=75, C=3, d_model=128):
        super().__init__()
        self.joint_embed = nn.Linear(C, 16)
        self.temporal_proj = nn.Linear(16 * V, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=T)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, 2)
        self.pool = TemporalAttentionPooling(d_model)
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        B, C, T, V = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.joint_embed(x)
        x = x.reshape(B, T, -1)
        x = self.temporal_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = self.pool(x)
        return self.fc(x)


# =========================================================
# 2. PREPROCESSING LOGIC
# =========================================================
class Preprocessor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        # We initialize instances inside the thread, but this class holds utility methods
        pass

    def extract_xy(self, results):
        """Extracts (75, 3) keypoints: Pose(33) + LH(21) + RH(21)"""
        # Return None if no hands are detected (Optional: strict filtering)
        # For better UX, we can return pose even if hands are missing,
        # but for WLASL, hands are critical.
        if results.left_hand_landmarks is None and results.right_hand_landmarks is None:
            return None

        def _get(lms, n):
            if lms is None:
                return np.zeros((n, 3), dtype=np.float32)
            return np.array(
                [[lm.x, lm.y, lm.z] for lm in lms.landmark], dtype=np.float32
            )

        pose = _get(results.pose_landmarks, 33)
        lh = _get(results.left_hand_landmarks, 21)
        rh = _get(results.right_hand_landmarks, 21)

        return np.concatenate([pose, lh, rh], axis=0)

    def sample_and_pad(self, frames, target_frames=32):
        """Resamples or pads frames to target length."""
        frames = np.array(frames)  # (T, 75, 3)
        num_frames = frames.shape[0]

        if num_frames >= target_frames:
            indices = np.linspace(0, num_frames - 1, target_frames).astype(np.int32)
            output = frames[indices]
        else:
            last_frame = frames[-1][np.newaxis, :, :]
            pad = np.repeat(last_frame, target_frames - num_frames, axis=0)
            output = np.concatenate([frames, pad], axis=0)

        output = output.transpose(2, 0, 1)  # (3, 32, 75)
        tensor = torch.FloatTensor(output).unsqueeze(0)  # (1, 3, 32, 75)
        return tensor


# =========================================================
# 3. OPTIMIZED VIDEO PROCESSOR (For WebRTC)
# =========================================================
class VideoProcessor:
    def __init__(self):
        # Attributes set via factory
        self.model = None
        self.device = None
        self.labels = None
        self.preprocessor = None
        self.threshold = 0.6
        self.frame_skip = 3

        # Components
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,  # Speed optimized
        )

        # State
        self.sequence = deque(maxlen=32)
        self.prediction_history = deque(maxlen=5)  # For voting
        self.frame_counter = 0
        self.current_prediction = "Waiting..."
        self.current_conf = 0.0

        # Thread-safe Communication
        self.result_queue = queue.Queue()

    def draw_hud(self, image, label, conf):
        h, w, _ = image.shape
        # Top Bar
        cv2.rectangle(image, (0, 0), (w, 60), (0, 0, 0), -1)
        # Confidence Line
        bar_width = int(w * conf)
        cv2.rectangle(image, (0, 55), (bar_width, 60), (0, 255, 0), -1)
        # Text
        color = (0, 255, 0) if conf > self.threshold else (150, 150, 150)
        text = f"{label} ({conf * 100:.0f}%)"
        cv2.putText(image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        self.frame_counter += 1

        # 1. MediaPipe (Run every frame for smooth drawing)
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)
        image.flags.writeable = True

        # Draw Skeleton
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS
        )

        # 2. Extract & Buffer
        if self.preprocessor:
            kp = self.preprocessor.extract_xy(results)
            if kp is not None:
                self.sequence.append(kp)

        # 3. Inference (Every N frames)
        if (
            self.model
            and len(self.sequence) == 32
            and (self.frame_counter % self.frame_skip == 0)
        ):
            try:
                tensor_kps = self.preprocessor.sample_and_pad(list(self.sequence)).to(
                    self.device
                )
                with torch.no_grad():
                    outputs = self.model(tensor_kps)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred_idx = torch.max(probs, 1)

                    this_conf = conf.item()
                    this_label = self.labels[pred_idx.item()]

                    # Voting System
                    if this_conf > self.threshold:
                        self.prediction_history.append(this_label)
                    else:
                        self.prediction_history.append("...")

                    most_common = Counter(self.prediction_history).most_common(1)
                    if most_common:
                        final_label, count = most_common[0]
                        if count >= 3:  # Majority consensus
                            self.current_prediction = final_label
                            self.current_conf = this_conf

                    # Send to UI
                    if not self.result_queue.full():
                        self.result_queue.put_nowait(
                            {
                                "label": self.current_prediction,
                                "conf": self.current_conf,
                                "probs": probs.cpu().numpy()[0],
                            }
                        )
            except Exception as e:
                print(f"Inference Error: {e}")

        # 4. Draw HUD
        self.draw_hud(image, self.current_prediction, self.current_conf)

        return av.VideoFrame.from_ndarray(image, format="bgr24")


# =========================================================
# 4. RESOURCE LOADING
# =========================================================
@st.cache_resource
def load_resources():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- UPDATE PATHS HERE ---
    # Ensure these point to your correct local files
    MODEL_PATH = "checkpoints/best_model_kps_transformer.pth"
    GLOSS_PATH = "data/wlasl_reduced/gloss_map.json"

    # Load Labels (Convert JSON list/dict to {0: "book", 1: "drink"})
    try:
        with open(GLOSS_PATH, "r") as f:
            raw_glosses = json.load(f)

        if isinstance(raw_glosses, list):
            labels = {i: gloss for i, gloss in enumerate(raw_glosses)}
        else:
            # Sort dict by value if it's {"book": 0}
            sorted_pairs = sorted(raw_glosses.items(), key=lambda x: x[1])
            labels = (
                {v: k for k, v in sorted_pairs}
                if isinstance(list(raw_glosses.values())[0], int)
                else {
                    int(k): v
                    for k, v in sorted(raw_glosses.items(), key=lambda x: int(x[0]))
                }
            )

    except FileNotFoundError:
        st.error(f"Gloss map not found at {GLOSS_PATH}")
        return None, None, None

    # Load Model
    try:
        model = TransformerEncoderKeypoints(num_classes=len(labels))
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
    except FileNotFoundError:
        st.error(f"Model checkpoint not found at {MODEL_PATH}")
        return None, None, None

    return model, labels, device


# =========================================================
# 5. MAIN APP
# =========================================================
st.set_page_config(page_title="WLASL Sign Recognition", layout="wide")
st.title("🤟 WLASL-35 Sign Language Recognizer")

# Load Resources
model, labels, device = load_resources()
preprocessor = Preprocessor()

if not model:
    st.stop()

# Sidebar
st.sidebar.title("Settings")
mode = st.sidebar.radio("Mode", ["Video Upload", "Live Webcam"])
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.6)

# --- VIDEO UPLOAD MODE ---
if mode == "Video Upload":
    st.subheader("Video File Analysis")
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        st_video = st.empty()
        progress_bar = st.progress(0)

        # 1. Process whole video
        valid_frames = []
        original_frames = []

        # MediaPipe instance for file processing
        holistic = mp.solutions.holistic.Holistic(static_image_mode=False)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        curr = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            original_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Extract
            frame.flags.writeable = False
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            kp = preprocessor.extract_xy(results)
            if kp is not None:
                valid_frames.append(kp)

            curr += 1
            if total_frames > 0:
                progress_bar.progress(min(curr / total_frames, 1.0))

        holistic.close()
        cap.release()

        # 2. Inference
        if len(valid_frames) > 0:
            tensor_in = preprocessor.sample_and_pad(valid_frames).to(device)
            with torch.no_grad():
                outputs = model(tensor_in)
                probs = torch.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probs, 1)

            pred_gloss = labels[pred_idx.item()]
            pred_conf = conf.item()

            st.success(f"Prediction: **{pred_gloss}** ({pred_conf * 100:.1f}%)")

            # 3. Playback with Overlay
            # Simple replay
            for frame in original_frames:
                frame = cv2.resize(frame, (640, 480))
                cv2.putText(
                    frame,
                    f"{pred_gloss} ({pred_conf * 100:.0f}%)",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                st_video.image(frame)
        else:
            st.error("No hands detected in video.")

# --- LIVE WEBCAM MODE ---
elif mode == "Live Webcam":
    st.subheader("Live Webcam Inference")
    col1, col2 = st.columns([3, 1])

    with col1:
        # Factory to inject loaded model into WebRTC processor
        def video_frame_callback_factory():
            processor = VideoProcessor()
            processor.model = model
            processor.device = device
            processor.labels = labels
            processor.preprocessor = preprocessor
            processor.threshold = threshold
            return processor

        ctx = webrtc_streamer(
            key="wlasl-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            video_processor_factory=video_frame_callback_factory,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col2:
        st.markdown("### Live Stats")
        ph_label = st.empty()
        ph_conf = st.empty()
        ph_chart = st.empty()

    # Queue Polling Loop (Does not rerun script)
    if ctx.state.playing:
        while True:
            try:
                if ctx.video_processor:
                    result = ctx.video_processor.result_queue.get(timeout=1.0)
                    ph_label.metric("Prediction", result["label"])
                    ph_conf.progress(
                        result["conf"], text=f"Confidence: {result['conf'] * 100:.0f}%"
                    )

                    # Chart
                    probs = result["probs"]
                    top5 = probs.argsort()[-5:][::-1]
                    chart_data = {labels[i]: probs[i] for i in top5}
                    ph_chart.bar_chart(chart_data)
            except queue.Empty:
                pass
            except Exception:
                break
            time.sleep(0.05)
