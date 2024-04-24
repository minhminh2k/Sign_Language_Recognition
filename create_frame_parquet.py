import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def create_parquet_landmark_from_video(results, frame, xyz):
    coordinates_xyz = xyz[['type', 'landmark_index']].drop_duplicates().reset_index(drop=True).copy()
    face = pd.DataFrame()
    pose = pd.DataFrame() 
    left_hand = pd.DataFrame() 
    right_hand = pd.DataFrame() 
    if results.face_landmarks:
        for i, point in enumerate(results.face_landmarks.landmark): 
            face.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z] 
    if results.pose_landmarks:
        for i, point in enumerate(results.pose_landmarks.landmark): 
            pose.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.left_hand_landmarks:
        for i, point in enumerate(results.left_hand_landmarks.landmark): 
            left_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z] 
    if results.right_hand_landmarks:
        for i, point in enumerate(results.right_hand_landmarks.landmark): 
            right_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    face = face.reset_index() \
        .rename(columns={'index': 'landmark_index'}) \
        .assign(type='face')
    pose = pose.reset_index() \
        .rename(columns={'index': 'landmark_index'}) \
        .assign(type='pose')
    left_hand = left_hand.reset_index() \
        .rename(columns={'index': 'landmark_index'}) \
        .assign(type='left_hand')
    right_hand = right_hand.reset_index() \
        .rename(columns={'index': 'landmark_index'}) \
        .assign(type='right_hand')
    
    landmarks = pd.concat([face, pose, left_hand, right_hand]).reset_index(drop=True)
    landmarks = coordinates_xyz.merge(landmarks, on=['type', 'landmark_index'], how='left')
    landmarks = landmarks.assign(frame=frame)

    return landmarks

def video_capture_loop(xyz, video_file):
    all_landmarks = []

    cap = cv2.VideoCapture(video_file)  

    with mp_holistic.Holistic(min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        frame = 0
        while cap.isOpened():
            frame += 1
            
            ret, image = cap.read()
            
            if not ret:
                break
                
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            landmarks = create_parquet_landmark_from_video(results, frame, xyz)
            all_landmarks.append(landmarks)
        
    return all_landmarks 

def create_output_parquet(video_path="WIN_20240424_01_43_53_Pro.mp4"):
    pq_file = 'resources/data/ISLR/train_landmark_files/1004012212.parquet'
    video_file = video_path  # Video file path
    xyz = pd.read_parquet(pq_file)
    landmarks = video_capture_loop(xyz, video_file)
    
    if landmarks:
        landmarks = pd.concat(landmarks).reset_index(drop=True)
        landmarks.to_parquet('resources/data/ISLR/output.parquet')
    else:
        print("No landmarks data to save.")

if __name__ == "__main__":
    # Read parquet files
    pq_file = 'resources/data/ISLR/train_landmark_files/1004012212.parquet'
    video_file = 'WIN_20240424_01_43_53_Pro.mp4'
    xyz = pd.read_parquet(pq_file)
    
    landmarks = video_capture_loop(xyz, video_file)
    # print(landmarks)
    
    if landmarks:
        landmarks = pd.concat(landmarks).reset_index(drop=True)
        landmarks.to_parquet('resources/data/ISLR/output.parquet')
    else:
        print("No landmarks data to save.")
             