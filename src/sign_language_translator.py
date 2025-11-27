import cv2
import numpy as np
import torch
import constants
import streamlit as st
from streamlit_webrtc import VideoProcessorBase
from my_model import MyModel
from feature_extractor import FeatureExtractor
from utils import draw_landmarks
from mock_model import MockModel

USE_MOCK_MODEL = True

# Run once when app starts
@st.cache_resource(show_spinner=False)
def load_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if USE_MOCK_MODEL:
        model = MockModel()
        model.to(device)
        model.eval()
    else:
        try:
            model = MyModel()
            state_dict = torch.load(constants.WEIGHT_PATH, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None

    return model, device

cached_model, cached_device = load_resources()

class SignLanguageTranslator(VideoProcessorBase):
    def __init__(self):
        self.device = cached_device
        self.model = cached_model

        self.extractor = FeatureExtractor()
        self.frame_sequence = []
        self.predictions = [] # Predicted class indices
        self.sentence = [] # Predicted glosses for display

    def __del__(self):
        # Clean up MediaPipe
        if hasattr(self, "extractor"):
            self.extractor.close()

    # Process and return each frame
    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")

        # --- Extract keypoints ---
        landmarks_results, keypoints = self.extractor.process_frame(img_bgr)

        # --- Visualize landmarks ---
        img_bgr.flags.writeable = True
        draw_landmarks(img_bgr, landmarks_results)

        # --- Prediction ---
        self.frame_sequence.append(keypoints)

        # Sliding window ([0..FRAME_SEQUENCE_LENGTH-1]->[1..FRAME_SEQUENCE_LENGTH]->...)
        self.frame_sequence = self.frame_sequence[-constants.FRAME_SEQUENCE_LENGTH:] # Keep the last FRAME_SEQUENCE_LENGTH keypoints

        if len(self.frame_sequence) == constants.FRAME_SEQUENCE_LENGTH:
            if self.model:
                sequence_array = np.array(self.frame_sequence)
                # Dim: [1, seq length, total keypoints]
                input_tensor = torch.tensor(sequence_array).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(input_tensor)

                probs = torch.softmax(logits, dim=1)
                predicted_prob_tensor, predicted_class_tensor = torch.max(probs, dim=1)

                predicted_prob = predicted_prob_tensor.item()
                predicted_class_idx = predicted_class_tensor.item()

                self.predictions.append(predicted_class_idx)

                print(f"Predicted prob: {predicted_prob}")
                print(f"Predicted class index: {predicted_class_idx}")

                if len(self.predictions) >= 10:
                    unique_pred_list = np.unique(self.predictions[-10:])

                    # Check if the last 10 predictions are consistent
                    if len(unique_pred_list) == 1 and unique_pred_list[0] == predicted_class_idx:
                        if predicted_prob > constants.PREDICTION_THRESHOLD:
                            current_gloss = list(constants.TRAINED_GLOSSES)[predicted_class_idx]
                            print(f"Predicted gloss: {current_gloss}")

                            if len(self.sentence) > 0:
                                if current_gloss != self.sentence[-1]:
                                    self.sentence.append(current_gloss)
                            else:
                                self.sentence.append(current_gloss)

                # Keep the last 5 glosses
                if len(self.sentence) > 5:
                    self.sentence = self.sentence[-5:]

                # Display predicted gloss on screen
                height, width, _ = img_bgr.shape
                cv2.rectangle(img_bgr, (0, height - 40), (width, height), (0, 0, 0), -1)

                text_to_show = " ".join(self.sentence)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2

                (text_w, text_h), _ = cv2.getTextSize(text_to_show, font, font_scale, thickness)

                # Center the text
                text_x = int((width - text_w) / 2)
                text_y = height - 10

                cv2.putText(img_bgr, text_to_show,
                            (text_x, text_y),
                            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return frame.from_ndarray(img_bgr, format="bgr24")