import sys
import os

sys.path.append(os.path.abspath("."))

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os
from collections import deque

from src.emotion.predictor import predict_emotion

st.set_page_config(page_title="Emotion AI", layout="wide")

st.title("🧠 Emotion Detection System")
st.write("Custom-trained CNN model for facial emotion recognition")

# 🔥 Deployment toggle
DEPLOY_MODE = os.environ.get("DEPLOY", "false") == "true"

# Sidebar
modes = ["Image Upload"]
if not DEPLOY_MODE:
    modes.append("Webcam")

mode = st.sidebar.radio("Choose Mode", modes)

st.sidebar.markdown("### About")
st.sidebar.write("CNN model trained on FER dataset")

# ---------------- IMAGE MODE ----------------
if mode == "Image Upload":

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        emotion, confidence, face_box = predict_emotion(image_np)

        if emotion and face_box:
            x, y, w, h = face_box

            cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 2)

            st.image(image_np, use_container_width=True)

            st.success(f"🎭 Emotion: {emotion.upper()}")
            st.progress(min(int(confidence), 100))

        else:
            st.warning("No face detected")


# ---------------- WEBCAM MODE ----------------
elif mode == "Webcam":

    if "run" not in st.session_state:
        st.session_state.run = False

    if "history" not in st.session_state:
        st.session_state.history = deque(maxlen=5)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Webcam"):
            st.session_state.run = True

    with col2:
        if st.button("Stop Webcam"):
            st.session_state.run = False

    FRAME_WINDOW = st.image([])

    if st.session_state.run:
        cap = cv2.VideoCapture(0)

        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not working")
                break

            emotion, confidence, face_box = predict_emotion(frame)

            if emotion:
                st.session_state.history.append(emotion)

                # 🔥 smoothing
                smooth_emotion = max(
                    set(st.session_state.history),
                    key=st.session_state.history.count
                )

                if face_box:
                    x, y, w, h = face_box

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    cv2.putText(
                        frame,
                        f"{smooth_emotion.upper()} ({confidence:.1f}%)",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)

            time.sleep(0.03)

        cap.release()