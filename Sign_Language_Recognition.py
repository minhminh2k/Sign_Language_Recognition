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
from model import KeyPointClassifier
from model import PointHistoryClassifier
import playsound
import sys
import datetime
import json
import re

import queue
from time import strftime
from utils_infor import *
from text_to_speech import speak_english, speak_vietnamese, translate_to_vn, reading_thread

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

# # Camera preparation ###############################################################
# cap = cv2.VideoCapture(cap_device)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

# Model load #############################################################
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=use_static_image_mode,
    max_num_hands=2,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence,
)

keypoint_classifier = KeyPointClassifier()

point_history_classifier = PointHistoryClassifier()

# Read labels ###########################################################
with open('model/keypoint_classifier/keypoint_classifier_label.csv',
            encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]
with open(
        'model/point_history_classifier/point_history_classifier_label.csv',
        encoding='utf-8-sig') as f:
    point_history_classifier_labels = csv.reader(f)
    point_history_classifier_labels = [
        row[0] for row in point_history_classifier_labels
    ]

# FPS Measurement ########################################################
cvFpsCalc = CvFpsCalc(buffer_len=10)

# Coordinate history #################################################################
history_length = 16
point_history = deque(maxlen=history_length)

# Finger gesture history ################################################
finger_gesture_history = deque(maxlen=history_length)

#  ########################################################################
mode = 0


# Demo video
DEMO_VIDEO = 'demo.mp4'
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
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
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
    # record = st.sidebar.checkbox("Record Video")
    # if record:
    #     st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    # sameer=""
    st.markdown(' ## Output')
    # st.markdown(sameer)

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

    # codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    # out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4
    frame_placeholder = st.empty()
    
    # Stop button
    stop_button_pressed = st.button("Stop")
    
    while True and not stop_button_pressed:
        fps = cvFpsCalc.get()
        
        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27 or stop_button_pressed:  # ESC
            break
        # number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            st.write("Video Capture Ended")
            break
        if use_webcam:
            image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                # logging_csv(number, mode, pre_processed_landmark_list,
                #             pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                
            
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    # vietnamese_text,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )

        else:
            point_history.append([0, 0])
        
        # if record:
        #     out.write(debug_image)
            
        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number=0)
        frame_placeholder.image(debug_image,channels="BGR", use_column_width=True)

    # st.text('Video Processed')

    # output_video = open('output1.mp4', 'rb')
    # out_bytes = output_video.read()
    # st.video(out_bytes)

    cap.release()
    # out.release()
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
