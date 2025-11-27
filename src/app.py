import streamlit as st
import constants
from streamlit_webrtc import webrtc_streamer
from sign_language_translator import SignLanguageTranslator

st.set_page_config(layout="wide")
st.title("Real-Time Sign Language Translator")

col1, col2 = st.columns([2, 1])

with col1:
    st.info(f"Please allow webcam access. This application will gather {constants.FRAME_SEQUENCE_LENGTH} consecutive frames and recognize what sign is being performed.")
    webrtc_streamer(
        key="sign-language-translator",
        video_processor_factory=SignLanguageTranslator,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("Model Details")
    st.write(f"**Trained Glosses:** {', '.join(constants.TRAINED_GLOSSES)}")
    st.write(f"**Sequence Length:** {constants.FRAME_SEQUENCE_LENGTH} frames")
    st.write(f"**Confidence Threshold:** {constants.PREDICTION_THRESHOLD * 100}%")
    st.markdown("---")
    st.subheader("How it Works")
    st.markdown("---")