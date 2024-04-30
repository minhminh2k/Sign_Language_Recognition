import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tempfile
import time
from PIL import Image, ImageTk
import os
import time
import csv
import copy
import argparse
import itertools
from collections import Counter, deque
import speech_recognition as sr
import json
from time import strftime
from moviepy.editor import VideoFileClip

from text_to_speech import translate_to_vn, speak_from_text, translate_from_en

# Isolated ASL
from inference import inference_ASL, load_relevant_data_subset
from create_frame_parquet import create_output_parquet

# Fingerspelling ASL
from inference_fingerspelling import load_relevant_data_subset_fingerspelling
from create_parquet_fingerspelling import create_output_parquet_fingerspelling


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

st.set_page_config(page_title="Sign Language Recognition", page_icon=":hand:")


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

# CSS styles
css = open("style.css")
st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)
css.close()

### Google - Isolated American Sign Language 
@st.cache_resource
def loading_model():
    interpreter = tf.lite.Interpreter(model_path="resources/model_weights/ISLR/model.tflite")
    found_signatures = list(interpreter.get_signature_list().keys())

    # if REQUIRED_SIGNATURE not in found_signatures:
    #     raise KernelEvalException('Required input signature not found.')

    prediction_fn = interpreter.get_signature_runner("serving_default")
    return prediction_fn

@st.cache_data
def loading_class():
    train = pd.read_csv('resources/data/ISLR/train.csv')
    train['sign_ord'] = train['sign'].astype('category').cat.codes

    # Dictionaries
    SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
    ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()
    return ORD2SIGN

def loading_inference_video(video_path="videos/hello.mp4", prediction_fn = None, ORD2SIGN = None):
    if prediction_fn is None:
        prediction_fn = loading_model()
        
    # Create output parquet file
    create_output_parquet(video_path)

    # Predict from parquet file
    pq_file = "resources/data/ISLR/output.parquet"

    try:
        coordinates_xyz_np = load_relevant_data_subset(pq_file)
        prediction = prediction_fn(inputs=coordinates_xyz_np)
        sign = np.argmax(prediction['outputs'])
        return ORD2SIGN[sign]
        # print(sign)
        
    except:
        return "Video is invalid!!!"

### Loading model and class ISL
prediction_fn = loading_model()
ORD2SIGN = loading_class()


### Google - American Sign Language Fingerspelling Recognition
REQUIRED_OUTPUT = "outputs"

@st.cache_resource
def loading_model_fingerspelling():
    model_path = "resources/model_weights/ASL_Fingerspelling/model.tflite"

    interpreter = tf.lite.Interpreter(model_path)
    found_signatures = list(interpreter.get_signature_list().keys())

    # if REQUIRED_SIGNATURE not in found_signatures:
    #     raise KernelEvalException('Required input signature not found.')

    prediction_fn = interpreter.get_signature_runner("serving_default")
    
    return prediction_fn

@st.cache_data
def loading_charater_fingerspelling():
    with open ("resources/data/ASL_Fingerspelling/character_to_prediction_index.json", "r") as f:
        character_map = json.load(f)
    rev_character_map = {j:i for i,j in character_map.items()}
    return rev_character_map

@st.cache_data
def loading_fingerspelling_columns():
    with open('resources/data/ASL_Fingerspelling/inference_args.json', 'r') as file:
        data = json.load(file)

    SEL_COLS = data['selected_columns']
    return SEL_COLS

def loading_inference_video_fingerspelling(video_path="videos/fingerspelling/obrien.mp4", prediction_fn=None, rev_character_map=None):
    # Create output.parquet
    create_output_parquet_fingerspelling(video_path)
    
    try:
        rq_path = 'resources/data/ASL_Fingerspelling/output.parquet'
        frames = load_relevant_data_subset_fingerspelling(rq_path)
        # print(frames.dtypes)
        frames = frames.astype('float32')

        output = prediction_fn(inputs=frames)
        prediction_str = "".join([rev_character_map.get(s, "") for s in np.argmax(output[REQUIRED_OUTPUT], axis=1)])
        return prediction_str
    except Exception as e:
        return e

### Loading model and class ASL Fingerspelling
prediction_fn_fingerspelling = loading_model_fingerspelling()
rev_character_map = loading_charater_fingerspelling()
SEL_COLS = loading_fingerspelling_columns()

### Page title
st.sidebar.title('Sign Language Recognition')

### App Mode
app_mode = st.sidebar.selectbox('Choose the App mode',
['About App','Isolated Sign Language Recognition', 'ASL Fingerspelling Recognition', 'Text to sign Language']
)

if app_mode =='About App':
    st.title('Sign Language Recognition')
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

elif app_mode == 'Isolated Sign Language Recognition':
    
    st.title(':call_me_hand: Isolated Sign Language Recognition from the video')
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
    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    st.markdown("In this mode, you could upload a short video about the sign language you want to understand. Let's try it :hugging_face: !!!")
    
    st.subheader('Upload a video')
    video_file_buffer = st.file_uploader("", type="mp4", key="ISL")

    if video_file_buffer is not None:
        # Check video length
        with open("videos/temp_video.mp4", "wb") as f:
            f.write(video_file_buffer.getbuffer())

        # Read the Mp4 file
        video_path = "videos/temp_video.mp4"
        video = VideoFileClip(video_path)
        duration = video.duration
        video.close()

        # Xóa tệp tạm thời
        os.remove(video_path)

        if duration > 30:
            st.warning("Sorry, the video is too long (maximum 30 seconds).", icon="⚠️")
        else:
            video_bytes = video_file_buffer.read()
            
            video_path_save = 'videos/video.mp4'
            
            with open(video_path_save, 'wb') as f:
                f.write(video_bytes)

            st.success("The video has been uploaded successfully.", icon="✅")
            
            st.video(video_path_save, loop=True)
            
            if st.button("Predict"):
                status_placeholder = st.empty()
                with status_placeholder:
                    st.write('<div style="text-align:center;">Processing...</div>', unsafe_allow_html=True)
                # predicted = inference_ASL(video_path_save)
                predicted = loading_inference_video(video_path_save, prediction_fn, ORD2SIGN)
                with status_placeholder:
                    status_placeholder.empty()
                
                # Print
                print(predicted)
                st.text_area(label="", value=predicted, height=50)
    
elif app_mode == 'ASL Fingerspelling Recognition':
    st.title(':v: American Sign Language Fingerspelling Recognition from the video')
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
    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    st.markdown("In this mode, you could upload a short video about the finger-spelling sign language you want to know. Enjoy :smile: !!! ")
    
    st.subheader('Upload a video')
    video_file_buffer_fingerspelling = st.file_uploader("", type="mp4", key="FP")

    if video_file_buffer_fingerspelling is not None:
        # Check video length
        with open("videos/fingerspelling/temp_video.mp4", "wb") as f:
            f.write(video_file_buffer_fingerspelling.getbuffer())

        # Read the Mp4 file
        video_path = "videos/fingerspelling/temp_video.mp4"
        video = VideoFileClip(video_path)
        duration = video.duration
        video.close()

        # Xóa tệp tạm thời
        os.remove(video_path)

        if duration > 30:
            st.warning("Sorry, the video is too long (maximum 30 seconds).", icon="⚠️")
        else:
            video_bytes = video_file_buffer_fingerspelling.read()
            
            video_path_save_fingerspelling = 'videos/fingerspelling/video.mp4'
            
            with open(video_path_save_fingerspelling, 'wb') as f:
                f.write(video_bytes)

            st.success("The video has been uploaded successfully.", icon="✅")
            
            st.video(video_path_save_fingerspelling, loop=True)
            
            if st.button("Predict"):
                status_placeholder = st.empty()
                with status_placeholder:
                    st.write('<div style="text-align:center;">Processing...</div>', unsafe_allow_html=True)

                predicted = loading_inference_video_fingerspelling(video_path_save_fingerspelling, prediction_fn_fingerspelling, rev_character_map)
                with status_placeholder:
                    status_placeholder.empty()
                
                # Print
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