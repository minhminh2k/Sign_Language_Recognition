import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# SEL_COLS = pd.read_parquet('105143404.parquet').columns[1:].tolist()

with open('resources/data/ASL_Fingerspelling/inference_args.json', 'r') as file:
    data = json.load(file)

SEL_COLS = data['selected_columns']

def create_parquet_landmark_from_video(results, frame):
    
    df = pd.DataFrame(columns=SEL_COLS)
    
    if results.face_landmarks:
        for i, point in enumerate(results.face_landmarks.landmark): 
            # print(type(point.x))
            df.loc[0, f"x_face_{i}"] = point.x
            df.loc[0, f"y_face_{i}"] = point.y
            df.loc[0, f"z_face_{i}"] = point.z
            
    if results.pose_landmarks:
        for i, point in enumerate(results.pose_landmarks.landmark): 
            df.loc[0, f"x_pose_{i}"] = point.x
            df.loc[0, f"y_pose_{i}"] = point.y
            df.loc[0, f"z_pose_{i}"] = point.z
    if results.left_hand_landmarks:
        for i, point in enumerate(results.left_hand_landmarks.landmark): 
            df.loc[0, f"x_left_hand_{i}"] = point.x
            df.loc[0, f"y_left_hand_{i}"] = point.y
            df.loc[0, f"z_left_hand_{i}"] = point.z
    if results.right_hand_landmarks:
        for i, point in enumerate(results.right_hand_landmarks.landmark): 
            df.loc[0, f"x_right_hand_{i}"] = point.x
            df.loc[0, f"y_right_hand_{i}"] = point.y
            df.loc[0, f"z_right_hand_{i}"] = point.z
            
    
    df = df.reset_index(drop=True).assign(frame=frame)
    frame_column = df['frame']

    df.drop(columns=['frame'], inplace=True)
    df.insert(0, 'frame', frame_column)
    
    return df

def video_capture_loop(video_file):
    all_landmarks = []

    cap = cv2.VideoCapture(video_file)  

    with mp_holistic.Holistic(min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        frame = 0
        while cap.isOpened():
            
            ret, image = cap.read()
            
            if not ret:
                break
                
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            landmarks = create_parquet_landmark_from_video(results, frame)
            all_landmarks.append(landmarks)
            
            frame += 1
            
        
    return all_landmarks 

def create_output_parquet_fingerspelling(video_path="videos/fingerspelling/obrien.mp4"):
    video_file = video_path  # Video file path
    landmarks = video_capture_loop(video_file)
    
    if landmarks:
        landmarks = pd.concat(landmarks).reset_index(drop=True)
        # print(landmarks)
        # landmarks.fillna(0, inplace=True)
        landmarks.to_parquet('resources/data/ASL_Fingerspelling/output.parquet')
    else:
        print("No landmarks data to save.")

if __name__ == "__main__":
    create_output_parquet_fingerspelling("videos/fingerspelling/obrien.mp4")
             