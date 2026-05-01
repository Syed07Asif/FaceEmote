import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.models import load_model

# 🔥 Cache model (faster loading)
@st.cache_resource
def load_emotion_model():
    return load_model("model/emotion_model.h5")

model = load_emotion_model()

EMOTIONS = ['angry','disgust','fear','happy','neutral','sad','surprise']

# Face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def predict_emotion(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(48, 48)
        )

        if len(faces) == 0:
            return None, None, None

        for (x, y, w, h) in faces:
            # Add padding for better context
            pad = 10
            face = gray[max(y-pad,0):y+h+pad, max(x-pad,0):x+w+pad]

            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = np.reshape(face, (1, 48, 48, 1))

            preds = model.predict(face, verbose=0)[0]

            max_index = np.argmax(preds)
            confidence = preds[max_index] * 100
            emotion = EMOTIONS[max_index]

            # Confidence filtering
            if confidence < 40:
                emotion = "neutral"

            return emotion, confidence, (x, y, w, h)

    except Exception as e:
        print("Prediction Error:", e)
        return None, None, None