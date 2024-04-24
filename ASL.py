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
import speech_recognition as sr
import json
from time import strftime
from text_to_speech import translate_to_vn, speak_from_text, translate_from_en

from inference import inference_ASL

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
                        
    args = parser.parse_args()

    return args
                    
args = get_args()

cap_device = args.device
cap_width = args.width
cap_height = args.height

use_static_image_mode = args.use_static_image_mode
min_detection_confidence = args.min_detection_confidence
min_tracking_confidence = args.min_tracking_confidence


# ----------------------Page config--------------------------------------

st.set_page_config(page_title="Isolated Sign Language Recognition", page_icon=":hand:")


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

css = open("style.css")
st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)
css.close()


st.sidebar.title('Sign Language Recognition')


app_mode = st.sidebar.selectbox('Choose the App mode',
['About App','Sign Language Recognition','Text to sign Language', 'Speech to sign Language']
)

if app_mode =='About App':
    st.title('Sign Language Detection')
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

elif app_mode == 'Sign Language Recognition':
    st.title(':call_me_hand: Sign Language Recognition from the video')
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
        textarea {
        font-size: 2rem !important;
        }
        .stButton > button {
        display: block;
        margin: 0 auto;
    }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # st.markdown(button_html, unsafe_allow_html=True)
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.sidebar.markdown('---')

    stframe = st.empty()
    
    # st.markdown("<p style='font-size: 24px; font-weight: bold;'>Upload a video</p>", unsafe_allow_html=True)
    st.subheader('Upload a video')
    video_file_buffer = st.file_uploader("", type="mp4")

    if video_file_buffer is not None:
        video_bytes = video_file_buffer.read()
        
        video_path_save = 'videos/video.mp4'
        
        with open(video_path_save, 'wb') as f:
            f.write(video_bytes)

        st.success("The video has been uploaded successfully.")
        
        st.video(video_path_save, loop=True)
        
        if st.button("Predict"):
            status_placeholder = st.empty()
            with status_placeholder:
                st.write('<div style="text-align:center;">Processing...</div>', unsafe_allow_html=True)
            predicted = inference_ASL(video_path_save)
            with status_placeholder:
                status_placeholder.empty()
            print(predicted)
            st.text_area(label="", value=predicted, height=50)
    
    
elif app_mode == 'Text to sign Language':
    st.title('Text to Sign Language')

    text = st.text_input("Enter text here:")
    # convert text to lowercase
    text = text.lower()
    
    if text:   
        # Translate to english and query
        input_text = translate_from_en(text)
        
        # if os.path.exists(f"videos/{input_text}.mp4"):
        #     st.video(f"videos/{input_text}.mp4", format="video/mp4") # support youtube video
        # else:
        #     st.write("No video file was found")
    
        st.video("https://qipedc.moet.gov.vn/videos/D0006.mp4?autoplay=true") # support youtube video

else:
    st.title('Speech to Sign Language')
    # initialize the speech recognition engine
    r = sr.Recognizer()

    # add start button to start recording audio
    if st.button("Start Talking"):
        # record audio for 5 seconds
        with sr.Microphone() as source:
            st.write("Say something!")
            audio = r.listen(source, phrase_time_limit=4)
            # st.write("Time over, thank you")
            try:
                text = r.recognize_google(audio, language="vi-VN")
            except sr.UnknownValueError:
                text = ""
                st.write("Sorry, I did not understand what you said.")
            
        if text != "":
            # convert text to lowercase
            text = text.lower()
            
            # display the final result
            st.write(f"You said: {text}", font_size=41)

            if text:   
                # Translate to english and query
                input_text = translate_from_en(text)
                
                if os.path.exists(f"videos/{input_text}.mp4"):
                    st.video(f"videos/{input_text}.mp4", format="video/mp4") # support youtube video
                else:
                    st.write("No video file was found")
                