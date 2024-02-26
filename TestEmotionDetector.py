import cv2
import numpy as np
import mediapipe as mp
from keras.models import model_from_json
import os
import pygame
import random
from datetime import datetime

# Define the path to the emotion model files
emotion_model_path = 'V:/PBL/AI/AI PBL/model/'
json_file_path = os.path.join(emotion_model_path, 'emotion_model.json')
weights_file_path = os.path.join(emotion_model_path, 'emotion_model.h5')

# Load json and create model
json_file = open(json_file_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load weights into the new model
emotion_model.load_weights(weights_file_path)
print("Loaded model from disk")

# Define the emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Initialize Pygame for audio playback
pygame.mixer.init()

# Define the path to the folder containing music for each emotion
music_folder = 'V:/PBL/AI/AI PBL/Music/'  # Update this path to your music folder

# Initialize MediaPipe for face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# OpenCV video capture
cap = cv2.VideoCapture(0)

# Variables for capturing image once
capture_image = False
image_captured = False

# Define the path for the captured images folder
captured_images_folder = 'V:/PBL/AI/AI PBL/Captured_Images/'
os.makedirs(captured_images_folder, exist_ok=True)  # Ensure the folder exists or create it

# Face detection context
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
                roi_gray_frame = frame[y:y + h, x:x + w]

                if not roi_gray_frame.size:
                    continue  # Skip this frame if no face is detected

                # Convert the image to grayscale
                roi_gray_frame = cv2.cvtColor(roi_gray_frame, cv2.COLOR_BGR2GRAY)
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                # Predict the emotions
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                detected_emotion = emotion_dict[maxindex]

                # Play a random song from the corresponding emotion folder
                emotion_music_folder = os.path.join(music_folder, detected_emotion)

                if not pygame.mixer.music.get_busy():
                    songs = os.listdir(emotion_music_folder)
                    if songs:
                        song_to_play = os.path.join(emotion_music_folder, random.choice(songs))
                        pygame.mixer.music.load(song_to_play)
                        pygame.mixer.music.play()

                        # Capture image once when a face is detected and song starts
                        capture_image = True

                cv2.putText(frame, detected_emotion, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Capture image only once when capture_image is True
        if capture_image and not image_captured:
            # Generate a timestamp for the unique filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            
            # Save the image with its corresponding emotion and timestamp
            image_filename = f'captured_image_{detected_emotion.lower()}_{timestamp}.jpg'
            image_path = os.path.join(captured_images_folder, image_filename)
            cv2.imwrite(image_path, frame)
            
            print(f"Image Captured! Emotion: {detected_emotion}")
            print(f"Image saved as: {image_filename}")
            
            image_captured = True

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
