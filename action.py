import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import csv

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from utils import CvFpsCalc

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
    

def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def draw_bounding_boxes(image, results, res):
    # Draw bounding box for hands
    if results.right_hand_landmarks:
        x_min = min(int(lm.x * image.shape[1]) for lm in results.right_hand_landmarks.landmark) - 15
        x_max = max(int(lm.x * image.shape[1]) for lm in results.right_hand_landmarks.landmark) + 15
        y_min = min(int(lm.y * image.shape[0]) for lm in results.right_hand_landmarks.landmark) - 15
        y_max = max(int(lm.y * image.shape[0]) for lm in results.right_hand_landmarks.landmark) + 15
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 4)
        
        cv2.rectangle(image, (x_min, y_min - 25), (x_max, y_min), (0, 255, 255), cv2.FILLED)
        cv2.putText(image, "Right: " + actions[np.argmax(res)] , (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        
    if results.left_hand_landmarks:
        x_min = min(int(lm.x * image.shape[1]) for lm in results.left_hand_landmarks.landmark) - 15
        x_max = max(int(lm.x * image.shape[1]) for lm in results.left_hand_landmarks.landmark)  + 15
        y_min = min(int(lm.y * image.shape[0]) for lm in results.left_hand_landmarks.landmark) - 15
        y_max = max(int(lm.y * image.shape[0]) for lm in results.left_hand_landmarks.landmark) + 15
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 4)
        cv2.rectangle(image, (x_min, y_min - 25), (x_max, y_min), (0, 255, 255), cv2.FILLED)
        cv2.putText(image, "Left: " + actions[np.argmax(res)], (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return image
    
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    # return np.concatenate([pose, face, lh, rh])
    return np.concatenate([pose, lh, rh])

def draw_fps_info(image, fps):
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return image

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou', 'fuckyou'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Label map
label_map = {label:num for num, label in enumerate(actions)}

cvFpsCalc = CvFpsCalc(buffer_len=10)

def main():
    # Create model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) # 1662 if using face contours
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    
    # Load model weights
    model.load_weights('action.h5')
    
    # 1. New detection variables
    sequence = []
    sentence = []
    threshold = 0.8
    res = ''
    
    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            fps = cvFpsCalc.get()
            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            # draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            if results.right_hand_landmarks or results.left_hand_landmarks:
                keypoints = extract_keypoints(results)

                sequence.append(keypoints)
                sequence = sequence[-30:]
                
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    
                #3. Viz logic
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]

                    # Viz probabilities
                    # image = prob_viz(res, actions, image, colors)
                    
                if results.right_hand_landmarks:
                    
                    x_min = min(int(lm.x * image.shape[1]) for lm in results.right_hand_landmarks.landmark) - 15
                    x_max = max(int(lm.x * image.shape[1]) for lm in results.right_hand_landmarks.landmark) + 15
                    y_min = min(int(lm.y * image.shape[0]) for lm in results.right_hand_landmarks.landmark) - 15
                    y_max = max(int(lm.y * image.shape[0]) for lm in results.right_hand_landmarks.landmark) + 15
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 4)
                    
                    cv2.rectangle(image, (x_min, y_min - 25), (x_max, y_min), (0, 255, 255), cv2.FILLED)
                    cv2.putText(image, "Right: " + actions[np.argmax(res)] , (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                    
                if results.left_hand_landmarks:
                    x_min = min(int(lm.x * image.shape[1]) for lm in results.left_hand_landmarks.landmark) - 15
                    x_max = max(int(lm.x * image.shape[1]) for lm in results.left_hand_landmarks.landmark)  + 15
                    y_min = min(int(lm.y * image.shape[0]) for lm in results.left_hand_landmarks.landmark) - 15
                    y_max = max(int(lm.y * image.shape[0]) for lm in results.left_hand_landmarks.landmark) + 15
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 4)
                    cv2.rectangle(image, (x_min, y_min - 25), (x_max, y_min), (0, 255, 255), cv2.FILLED)
                    cv2.putText(image, "Left: " + actions[np.argmax(res)], (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            
            image = draw_fps_info(image, fps)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & cv2.waitKey(10) == 27: # Esc to release
                break
        cap.release()
        cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    main()