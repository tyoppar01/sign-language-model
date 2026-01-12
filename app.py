import streamlit as st
import cv2
import torch
import numpy as np
import mediapipe as mp
import json
import tempfile
from pathlib import Path
import time
import threading
from collections import deque

# Import models from the package
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
from sign_language_model.models import TransformerEncoderKeypoints

# =========================================================
# PREPROCESSING LOGIC
# =========================================================
class Preprocessor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )

    def extract_xy(self, results):
        """Extracts (75, 3) keypoints: Pose(33) + LH(21) + RH(21)"""
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

    def process_frame(self, frame):
        """Runs MediaPipe on a single frame."""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = self.holistic.process(img_rgb)
        return results

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
# VIDEO PROCESSOR WITH THREADING
# =========================================================
class VideoProcessor:
    def __init__(self, model, device, labels, preprocessor, threshold=0.5):
        self.model = model
        self.device = device
        self.labels = labels
        self.preprocessor = preprocessor
        self.threshold = threshold
        
        # Buffer for keypoints
        self.sequence = deque(maxlen=45)
        self.frame_count = 0
        
        # Threading for non-blocking inference
        self.detected_label = "Waiting..."
        self.confidence = 0.0
        self.top5_predictions = []
        self.lock = threading.Lock()
        self.is_processing = False

    def inference(self, sequence):
        """Run inference in background thread."""
        try:
            with torch.no_grad():
                tensor_kps = self.preprocessor.sample_and_pad(list(sequence)).to(self.device)
                outputs = self.model(tensor_kps)
                
                probs = torch.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probs, 1)
                top5_prob, top5_idx = torch.topk(probs, min(5, len(self.labels)))
                
                with self.lock:
                    self.confidence = float(conf.item())
                    if self.confidence > self.threshold:
                        self.detected_label = self.labels[pred_idx.item()]
                    else:
                        self.detected_label = "..."
                    
                    # Store top 5
                    self.top5_predictions = [
                        (self.labels[top5_idx[0][i].item()], top5_prob[0][i].item())
                        for i in range(len(top5_prob[0]))
                    ]
        except Exception:
            pass
        finally:
            self.is_processing = False

    def process_frame(self, frame):
        """Process a single frame and return annotated frame."""
        self.frame_count += 1
        
        # Extract keypoints
        results = self.preprocessor.process_frame(frame)
        kp = self.preprocessor.extract_xy(results)
        
        # Visualize skeleton
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp.solutions.holistic.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp.solutions.holistic.HAND_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
        )
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp.solutions.holistic.HAND_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
        
        # Add to buffer
        if kp is not None:
            self.sequence.append(kp)
        
        # Run inference if buffer is ready and not already processing
        if len(self.sequence) >= 16 and not self.is_processing:
            if results.left_hand_landmarks or results.right_hand_landmarks:
                self.is_processing = True
                thread = threading.Thread(target=self.inference, args=(self.sequence,))
                thread.daemon = True
                thread.start()
        
        # Get current predictions (thread-safe)
        with self.lock:
            current_label = self.detected_label
            current_conf = self.confidence
        
        # Draw prediction on frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        color = (0, 255, 0) if current_label != "Waiting..." and current_label != "..." else (128, 128, 128)
        cv2.putText(
            frame_rgb,
            current_label,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
        )
        
        return frame_rgb, current_label, current_conf, self.top5_predictions.copy()


# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource
def load_model(num_classes, device):
    """Load TransformerEncoderKeypoints model."""
    checkpoint_path = "checkpoints/best_model_kps_transformer.pth"
    model = TransformerEncoderKeypoints(num_classes=num_classes)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Checkpoint not found: {checkpoint_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# =========================================================
# STREAMLIT APP
# =========================================================
st.set_page_config(
    page_title="WLASL Sign Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load resources
@st.cache_resource
def load_gloss_map():
    """Load gloss map."""
    try:
        with open("data/wlasl_reduced/gloss_map.json", "r") as f:
            raw_glosses = json.load(f)
        labels = {i: gloss for i, gloss in enumerate(raw_glosses)}
        return labels
    except FileNotFoundError:
        st.error("Gloss map not found!")
        return None

labels = load_gloss_map()
if labels is None:
    st.stop()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(len(labels), device)

if model is None:
    st.error("Failed to load model. Please check the checkpoint file.")
    st.stop()

preprocessor = Preprocessor()

# Main header
st.markdown('<div class="main-header">🤟 WLASL-35 Sign Language Recognizer</div>', unsafe_allow_html=True)

st.markdown("---")

# Mode selection
mode = st.radio(
    "Select Input Mode",
    ["Video Upload", "Live Webcam"],
    horizontal=True,
    label_visibility="collapsed"
)

# =========================================================
# VIDEO UPLOAD MODE
# =========================================================
if mode == "Video Upload":
    st.subheader("📹 Video Upload")
    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a video file containing sign language gestures"
    )
    
    if uploaded_file and model:
        # Save uploaded file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Processing columns
        col_video, col_results = st.columns([2, 1])
        
        with col_video:
            st_video = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # Extract features
        valid_frames = []
        original_frames = []
        curr_frame = 0
        
        status_text.text("🔄 Processing video frames...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            original_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            results = preprocessor.process_frame(frame)
            kp = preprocessor.extract_xy(results)
            
            if kp is not None:
                valid_frames.append(kp)
            
            curr_frame += 1
            if frame_count > 0:
                progress_bar.progress(min(curr_frame / frame_count, 1.0))
        
        cap.release()
        progress_bar.empty()
        status_text.empty()
        
        if len(valid_frames) == 0:
            st.error("❌ No hands/pose detected in the video.")
        else:
            # Run inference
            status_text.text("🔄 Running inference...")
            
            with torch.no_grad():
                tensor_kps = preprocessor.sample_and_pad(valid_frames).to(device)
                outputs = model(tensor_kps)
            
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            top5_prob, top5_idx = torch.topk(probs, min(5, len(labels)))
            
            pred_gloss = labels[pred_idx.item()]
            pred_conf = conf.item()
            
            status_text.empty()
            
            # Display results
            with col_results:
                st.markdown("### 📊 Predictions")
                
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown(f"### {pred_gloss}")
                st.markdown(f"**Confidence:** {pred_conf * 100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("#### Top 5 Predictions")
                for i in range(len(top5_prob[0])):
                    p = top5_prob[0][i].item()
                    gloss_label = labels[top5_idx[0][i].item()]
                    st.progress(p, text=f"{gloss_label}: {p * 100:.1f}%")
            
            # Create video with overlay for replay
            with col_video:
                st.markdown("### 📹 Video Preview")
                
                video_frames_with_overlay = []
                for frame in original_frames:
                    frame_display = frame.copy()
                    frame_display = cv2.resize(frame_display, (640, 480))
                    
                    # Add text overlay
                    cv2.putText(
                        frame_display,
                        f"{pred_gloss} ({pred_conf * 100:.0f}%)",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    video_frames_with_overlay.append(frame_display)
                
                # Save video for replay - using imageio if available, otherwise OpenCV
                temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_video_path.close()
                
                video_written = False
                
                # Try imageio first (better compatibility)
                try:
                    import imageio
                    imageio.mimwrite(
                        temp_video_path.name,
                        video_frames_with_overlay,
                        fps=fps,
                        codec='libx264',
                        quality=8
                    )
                    if Path(temp_video_path.name).stat().st_size > 0:
                        video_written = True
                except ImportError:
                    pass
                except Exception:
                    pass
                
                # Fallback to OpenCV
                if not video_written:
                    width, height = 640, 480
                    codecs_to_try = [('XVID', 'XVID'), ('mp4v', 'MP4V'), ('avc1', 'H264')]
                    for fourcc_str, _ in codecs_to_try:
                        try:
                            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                            out = cv2.VideoWriter(temp_video_path.name, fourcc, fps, (width, height))
                            if not out.isOpened():
                                continue
                            for frame in video_frames_with_overlay:
                                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                out.write(frame_bgr)
                            out.release()
                            if Path(temp_video_path.name).stat().st_size > 0:
                                video_written = True
                                break
                        except Exception:
                            continue
                
                # Display video
                if video_written:
                    try:
                        with open(temp_video_path.name, 'rb') as video_file:
                            video_bytes = video_file.read()
                        if len(video_bytes) > 0:
                            st_video.video(video_bytes, format='video/mp4', start_time=0)
                        else:
                            raise ValueError("Video file is empty")
                    except Exception as e:
                        st.warning(f"Could not display video: {e}")
                        # Fallback to showing first frame
                        if video_frames_with_overlay:
                            st_video.image(video_frames_with_overlay[0], channels="RGB")
                else:
                    st.warning("Could not create video file. Showing first frame:")
                    if video_frames_with_overlay:
                        st_video.image(video_frames_with_overlay[0], channels="RGB")
        
        # Cleanup
        Path(tfile.name).unlink(missing_ok=True)

# =========================================================
# LIVE WEBCAM MODE
# =========================================================
elif mode == "Live Webcam":
    st.subheader("📷 Live Webcam Inference")
    
    with st.expander("📷 Camera Settings", expanded=False):
        camera_index = st.number_input(
            "Camera Index",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="Select camera index (0 for default webcam, 1 for second camera, etc.)"
        )
        
        threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    # Initialize processor
    if "video_processor" not in st.session_state:
        st.session_state.video_processor = VideoProcessor(
            model=model,
            device=device,
            labels=labels,
            preprocessor=preprocessor,
            threshold=threshold
        )
    
    processor = st.session_state.video_processor
    processor.threshold = threshold
    
    # Layout
    col_camera, col_live_results = st.columns([2, 1])
    
    with col_camera:
        st_frame = st.empty()
    
    with col_live_results:
        st_live_gloss = st.empty()
        st_live_conf = st.empty()
        st_live_chart = st.empty()
    
    # Camera control
    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False
    
    col_start, col_stop = st.columns([1, 1])
    with col_start:
        if st.button("🎥 Start Camera", use_container_width=True):
            st.session_state.camera_running = True
            st.rerun()
    
    with col_stop:
        if st.button("⏹️ Stop Camera", use_container_width=True):
            st.session_state.camera_running = False
            processor.sequence.clear()
            st.rerun()
    
    if st.session_state.camera_running:
        # Initialize or get camera
        if "webcam_cap" not in st.session_state:
            st.session_state.webcam_cap = None
        
        if st.session_state.webcam_cap is None:
            # Try to open camera
            cap = cv2.VideoCapture(int(camera_index))
            if cap.isOpened():
                # Set camera properties for better compatibility
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                st.session_state.webcam_cap = cap
            else:
                st.error(f"❌ Failed to open camera {int(camera_index)}. Please check if the camera is available and not being used by another application.")
                st.session_state.camera_running = False
                cap = None
        
        cap = st.session_state.webcam_cap
        
        if cap is not None and cap.isOpened():
            # Read and process one frame
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Camera failed to read frame")
                st.session_state.camera_running = False
            else:
                # Process frame (non-blocking inference via threading)
                frame_rgb, pred_gloss, pred_conf, top5_preds = processor.process_frame(frame)
                
                # Track last prediction to only update UI when it changes
                if "last_pred_gloss" not in st.session_state:
                    st.session_state.last_pred_gloss = None
                if "last_update_time" not in st.session_state:
                    st.session_state.last_update_time = time.time()
                
                # Only update UI if prediction changed or enough time passed (reduce jitter)
                current_time = time.time()
                prediction_changed = (pred_gloss != st.session_state.last_pred_gloss and 
                                    pred_gloss != "Waiting..." and 
                                    pred_gloss != "...")
                time_elapsed = current_time - st.session_state.last_update_time
                
                # Update display
                st_frame.image(frame_rgb, channels="RGB")
                
                # Only update metrics/charts if prediction changed or 0.5 seconds passed
                if prediction_changed or time_elapsed >= 0.5:
                    st_live_gloss.metric("Detected", pred_gloss)
                    st_live_conf.progress(pred_conf, text=f"Confidence: {pred_conf * 100:.1f}%")
                    
                    if top5_preds:
                        chart_data = {label: prob for label, prob in top5_preds}
                        st_live_chart.bar_chart(chart_data)
                    
                    st.session_state.last_pred_gloss = pred_gloss
                    st.session_state.last_update_time = current_time
                
                # Rerun less frequently - only every 0.3 seconds (3-4 FPS) to reduce jitter
                time.sleep(0.3)
                st.rerun()
        else:
            if cap is not None:
                cap.release()
                st.session_state.webcam_cap = None
    
    # Status info at bottom (small and grayed out)
    st.markdown("---")
    st.markdown(
        f'<div style="text-align: center; color: #888; font-size: 0.75rem; margin-top: 2rem;">'
        f'Device: {device.upper()} | Model: TransformerEncoder | Classes: {len(labels)}'
        f'</div>',
        unsafe_allow_html=True
    )
