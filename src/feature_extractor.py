import cv2
import constants
from mediapipe.python.solutions import holistic as mp_holistic
from utils import extract_keypoints

class FeatureExtractor:
    def __init__(self):
        # MediaPipe init
        self.mp_holistic = mp_holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=constants.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=constants.MIN_TRACKING_CONFIDENCE,
        )

    def close(self):
        self.holistic.close()

    def process_frame(self, frame_bgr):
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False

        landmarks_results = self.holistic.process(img_rgb)
        keypoints = extract_keypoints(landmarks_results) # type: ignore

        return landmarks_results, keypoints
