import numpy as np
import constants
from mediapipe.python.solutions import holistic as mp_holistic
from mediapipe.python.solutions import drawing_utils as mp_drawing

# TODO: Update if necessary
def extract_keypoints(landmarks):
    # Pose (Body)
    if landmarks.pose_landmarks:
        pose = np.array(
            [[res.x, res.y, res.z, res.visibility] for res in landmarks.pose_landmarks.landmark]
        ).flatten()
    else:
        pose = np.zeros(constants.POSE_INPUT_DIM)

    # Face
    if landmarks.face_landmarks:
        face = np.array(
            [[res.x, res.y, res.z] for res in landmarks.face_landmarks.landmark]
        ).flatten()
    else:
        face = np.zeros(constants.FACE_INPUT_DIM)

    # Left Hand
    if landmarks.left_hand_landmarks:
        lh = np.array(
            [[res.x, res.y, res.z] for res in landmarks.left_hand_landmarks.landmark]
        ).flatten()
    else:
        lh = np.zeros(constants.LEFT_HAND_INPUT_DIM)

    # Right Hand
    if landmarks.right_hand_landmarks:
        rh = np.array(
            [[res.x, res.y, res.z] for res in landmarks.right_hand_landmarks.landmark]
        ).flatten()
    else:
        rh = np.zeros(constants.RIGHT_HAND_INPUT_DIM)

    return np.concatenate([pose, face, lh, rh])


def draw_landmarks(img_bgr, landmarks_results):
    """
    img_bgr is modified in-place
    """

    # Draw face connections
    mp_drawing.draw_landmarks(
        img_bgr,
        landmarks_results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
    )

    # Draw pose connections
    mp_drawing.draw_landmarks(
        img_bgr,
        landmarks_results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
    )

    # Draw left hand connections
    mp_drawing.draw_landmarks(
        img_bgr,
        landmarks_results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )

    # Draw right hand connections
    mp_drawing.draw_landmarks(
        img_bgr,
        landmarks_results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )

    return img_bgr