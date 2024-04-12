import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image, ImageTk
import os
from PIL import Image
import time

import csv
import copy
import argparse
import itertools
from collections import Counter, deque
import threading

from utils import CvFpsCalc
import playsound
import sys
import datetime
import json
import re
import queue
from time import strftime
from text_to_speech import speak_english, speak_vietnamese, translate_to_vn, reading_thread
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from load_tensor_model import create_model, model_tensor

# Argument Parsers
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)
    parser.add_argument("--font",
                        help='font path',
                        type=str,
                        default='Arimo-Bold.ttf')
                        
    args = parser.parse_args()

    return args
                    
args = get_args()

cap_device = args.device
cap_width = args.width
cap_height = args.height

use_static_image_mode = args.use_static_image_mode
min_detection_confidence = args.min_detection_confidence
min_tracking_confidence = args.min_tracking_confidence

use_brect = True

# Model load #############################################################
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

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
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
    #                          mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
    #                          mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    #                          ) 
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
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def draw_fps_info(image, fps):
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return image

# FPS Measurement ########################################################
cvFpsCalc = CvFpsCalc(buffer_len=10)

# Demo video
DEMO_VIDEO = 'dqm.mp4'
DEMO_IMAGE = 'demo.jpg'

my_list = []

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Sign Language Recognition')
# st.sidebar.subheader('-Parameter')

@st.cache_data()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:

        r = height / float(h)
        dim = (int(w * r), height)


    else:

        r = width / float(w)
        dim = (width, int(h * r))


    resized = cv2.resize(image, dim, interpolation=inter)


    return resized

app_mode = st.sidebar.selectbox('Choose the App mode',
['About App','Sign Language to Text','Text to sign Language']
)

if app_mode =='About App':
    st.title('Sign Language Detection Using MediaPipe with Streamlit GUI')
    st.markdown('In this application we are using **MediaPipe** for detecting Sign Language. **SpeechRecognition** Library of python to recognise the voice and machine learning algorithm which convert speech to the Indian Sign Language .**StreamLit** is to create the Web Graphical User Interface (GUI) ')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    st.video('https://youtu.be/NYAFEmte4og')
    st.markdown('''
            # About Me \n 
                Hey this is **Sameer Edlabadkar**. Working on the technologies such as **Tensorflow, MediaPipe, OpenCV, ResNet50**. \n

                Also check me out on Social Media
                - [YouTube](https://www.youtube.com/@edlabadkarsameer/videos)
                - [LinkedIn](https://www.linkedin.com/in/sameer-edlabadkar-43b48b1a7/)
                - [GitHub](https://github.com/edlabadkarsameer)
            If you are facing any issue while working feel free to mail me on **edlabadkarsameer@gmail.com**

                ''')
elif app_mode == 'Sign Language to Text':
    st.title('Sign Language to Text')
    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Webcam')

    st.sidebar.markdown('---')
    st.markdown(' ## Output')

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            cap = cv2.VideoCapture(cap_device)
        else:
            cap = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO

    else:
        tfflie.write(video_file_buffer.read())
        cap = cv2.VideoCapture(tfflie.name)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))



    st.markdown("<hr/>", unsafe_allow_html=True)

    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 350px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 350px;
            margin-left: -350px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    sequence = []
    sentence = []
    threshold = 0.8
    res = ''

    frame_placeholder = st.empty()
    
    # Stop button
    stop_button_pressed = st.button("Stop")
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True and not stop_button_pressed:
            fps = cvFpsCalc.get()
            # Read feed
            ret, frame = cap.read()
            
            if not ret:
                st.write("Video Capture Ended")
                break
            
            # if use_webcam:
            #     image = cv2.flip(frame, 1)  # Mirror display
            
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
                    res = model_tensor.predict(np.expand_dims(sequence, axis=0))[0]
                    
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
            
            frame_placeholder.image(image,channels="BGR", use_column_width=True)

    cap.release()
    cv2.destroyAllWindows()
else:
    st.title('Text to Sign Language')


    # define function to display sign language images
    def display_images(text):
        # get the file path of the images directory
        img_dir = "images/"

        # initialize variable to track image position
        image_pos = st.empty()

        # iterate through the text and display sign language images
        for char in text:
            if char.isalpha():
                # display sign language image for the alphabet
                img_path = os.path.join(img_dir, f"{char}.png")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=500)

                # wait for 2 seconds before displaying the next image
                time.sleep(1)

                # remove the image
                image_pos.empty()
            elif char == ' ':
                # display space image for space character
                img_path = os.path.join(img_dir, "space.png")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=500)

                # wait for 2 seconds before displaying the next image
                time.sleep(1)

                # remove the image
                image_pos.empty()

        # wait for 2 seconds before removing the last image
        time.sleep(2)
        image_pos.empty()


    text = st.text_input("Enter text:")
    # convert text to lowercase
    text = text.lower()

    # display sign language images
    display_images(text)
