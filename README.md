FaceEmote 
🧠 Real-Time Facial Emotion Detection System

I’m glad to share that I’ve built a real-time facial emotion detection system using a custom-trained CNN model.

Instead of using a pre-trained model, I trained the model from scratch on the FER-2013 dataset to better understand how the full machine learning pipeline works.

🎥 Demo of Realtime Detection using webcam

👉 [https://drive.google.com/file/d/1-FmTBEdpd0Tl79LtJgsog_oEZJftY1U7/view?usp=drivesdk]

📸 Screenshots

[App Screenshot 1](https://drive.google.com/file/d/1PANLc-7or9VJ6EgC_Q8doxb_BdihYRrm/view?usp=drivesdk)
[App Screenshot 2](https://drive.google.com/file/d/1JUUJq2jGqTMP9RT2hFi6-FvhKTk2pylv/view?usp=drivesdk)

🚀 Features

- Real-time emotion detection using webcam
- Image-based emotion prediction
- Face detection using OpenCV
- Custom CNN model trained from scratch
- Smooth and simple UI using Streamlit

🧠 What I Learned

- Importance of data preprocessing and normalization  
- Challenges in real-time prediction  
- Model bias towards certain emotions  
- Difference between training accuracy and real-world performance  

🛠 Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV  
- Streamlit  
- NumPy  

⚠️ Challenges Faced

- Model initially predicted mostly "happy" and "neutral"  
- Handling real-time performance and lag  
- Ensuring consistent predictions across different lighting conditions  

📌 Future Improvements

- Improve model accuracy using better architectures  
- Add more balanced dataset  
- Optimize for deployment  
- Extend to video input and advanced analytics  

📂 Project Structure

FaceEmote/
│
├── model/
│   └── emotion_model.h5
│
├── src/
│   └── emotion/
│       └── predictor.py
│
├── app.py
├── requirements.txt

▶️ How to Run

```bash
**Clone repo**
git clone https://github.com/Syed07Asif/FaceEmote.git

**Go to folder**
cd FaceEmote

**Install dependencies**
pip install -r requirements.txt

**Run app**
streamlit run app.py

🤝 Connect with Me

* LinkedIn: https://www.linkedin.com/in/syed-asif-59b81728b


⭐ If you like this project, feel free to star the repo!
