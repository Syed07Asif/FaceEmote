import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("model/emotion_model.h5")

EMOTIONS = ['angry','disgust','fear','happy','neutral','sad','surprise']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def predict_emotion(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None, None, None

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]

            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = np.reshape(face, (1, 48, 48, 1))

            preds = model.predict(face, verbose=0)[0]

            emotion = EMOTIONS[np.argmax(preds)]
            confidence = np.max(preds) * 100

            return emotion, confidence, (x, y, w, h)

    except Exception as e:
        print("Prediction Error:", e)
        return None, None, None