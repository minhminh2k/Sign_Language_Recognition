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
import subprocess

from text_to_speech import translate_to_vn, speak_from_text, translate_from_en

# Isolated ASL
from inference import inference_ASL, load_relevant_data_subset
from create_frame_parquet import create_output_parquet

# Fingerspelling ASL
from inference_fingerspelling import load_relevant_data_subset_fingerspelling
from create_parquet_fingerspelling import create_output_parquet_fingerspelling
from condition_dictionary import getValue, changeValue, createCsv



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

st.set_page_config(page_title="Sign Language Recognition", page_icon=":hand:", layout="wide")


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
        return "Video is invalid !!!"
    
# Function to record video
def record_video(filename, duration, frame_window):
    cap = cv2.VideoCapture(0)  # Start the webcam
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define the codec
    out = cv2.VideoWriter(filename, fourcc, 24.0, (640, 480))  # Create VideoWriter object

    start_time = time.time()

    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        if ret:
            out.write(frame)  # Write the frame into the file
            # cv2.imshow('Recording...', frame)  # Display the recording frame
            frame_window.image(frame, channels='BGR')
        if (time.time() - start_time) > duration:  # Check if duration is exceeded
            break
        # if cv2.waitKey(1) & 0xFF == ord('q'):  # Stop recording on 'q' key
        #     break

    cap.release()  # Release the capture
    out.release()  # Release the writer
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Function to display a YouTube video within Streamlit
def show_video(link):
    st.video(link)

def run():
    st.session_state.run = True

### Loading model and class ASL Fingerspelling
prediction_fn_fingerspelling = loading_model_fingerspelling()
rev_character_map = loading_charater_fingerspelling()
SEL_COLS = loading_fingerspelling_columns()

### Page title
st.sidebar.title('Sign Language Recognition')

### App Mode
app_mode = st.sidebar.selectbox('Choose the App mode',
['About App','Isolated Sign Language Recognition', 'ASL Fingerspelling Recognition', 'Dictionary', 'Video Quiz', 'Action Checker']
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
    
    st.markdown("This mode allows uploading videos and getting the corresponding isolated sign language returned. Users can upload videos depicting sign language, and the system will then analyze and return the isolated sign language as text based on the content of the uploaded video. Let's try it :hugging_face: !!!")
    st.markdown("---")
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
    st.title(':v: American Sign Language Fingerspelling Recognition')
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
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size:1.1rem;
        }
        .font_des {
            font-size:15px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    upload_video, finger_data = st.tabs(["Upload a video", "ASL Fingerspelling"])
    
    with upload_video:
        st.markdown("")
        st.markdown("This mode allows uploading videos and getting the corresponding American sign language finger-spelling returned. Users can upload videos depicting sign language, and the system will then analyze and return the American sign language finger-spelling as text based on the content of the uploaded video. Enjoy :smile: !!!")
        st.markdown("---")
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

            if duration > 60:
                st.warning("Sorry, the video is too long (maximum 60 seconds).", icon="⚠️")
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
                    
    with finger_data: 
        st.markdown("")
        st.markdown('This image shows American Sign Language fingerspelling. The use of sign language is an important means of communication for deaf or hard of hearing people, helping them convey meaning and interact with the community around them.', unsafe_allow_html=True)
        st.markdown("")

        col1, col2, col3 = st.columns([0.6, 3.8, 0.6])
        col2.image('resources/data/ASL_Fingerspelling/ASL.png', caption='American Sign Language Fingerspelling', width = 300, use_column_width=True)

elif app_mode == 'Dictionary':
    st.title("Dictionary")
    st.markdown("This dictionary will contain videos of the corresponding sign language words. Hopefully these videos will help you learn and use sign language better.")
    st.markdown("If there is a video, a button will appear corresponding to the word you searched for. And you just need to click on that button and the video will be displayed. If there is no video, there will be a warning.")
    # Connect to the Google Sheet
    sheet_id = "1dbaXMziDDIQ9Rbt7yoNQPWMSOw72iGs1HNAYwPiu7lU"
    sheet_name = "dictionary"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    @st.cache_data
    def load_data():
        # Read data from Google Sheet
        df = pd.read_csv(url, dtype=str).fillna("")
        return df

    df = load_data()

    if 'stage' not in st.session_state:
        st.session_state.stage = 0
        createCsv()

    def set_stage(stage):
            st.session_state.stage = stage

    def clear_text():
        st.session_state.my_text = st.session_state.widget
        #print("Thay doi text_search")
        changeValue("Choose_label", "0")
        changeValue("Click", "0")
        changeValue("All", "0")
        changeValue("Search", "1")
        st.session_state.widget = ""

    text_search = st.text_input("Enter the word you want to search for in the search bar", value="", 
                key='widget', 
                on_change=clear_text
                ).lower()

    #print("Bat dau lai tu dau")

    text_search = st.session_state.get('my_text', '')


    if getValue("Search") == "0":
        text_search = ""
        
    #print("Text search is:", text_search)

    choose_label = getValue("Choose_label")
    #print("Choose label là:", choose_label)

    if choose_label == "0":
        #print("chay lai")
        print("")

    if choose_label != "0":
        text_search = choose_label
        #print("da chon")

    # Filter the dataframe based on search term
    m1 = df["Labels"].str.contains(text_search)
    df_search = df[m1]

    # Show warning if no matching results found
    if df_search.empty and text_search:
        st.warning("No matching results found for the search term.", icon="⚠️")

    # Show the cards
    N_cards_per_row = 4

    #print("GetAll: ", getValue("All"))

    if not text_search or getValue("All") == "1":
        df_display = df
    else:
        df_display = df_search

    #print("Bắt đầu lại")
    #print("session stage:" ,st.session_state.stage)
    #print("Click:", getValue("Click"))

    if getValue("Click") == "0":
        st.markdown(
            """
            <style>
                /* Thiết kế lại giao diện cho nút button */
                div.stButton > button {
                    padding: 0.8em 2em;
                    font-size: 8px;
                    text-transform: uppercase;
                    letter-spacing: 2.5px;
                    font-weight: 300;
                    color: #000;
                    background-color: #fff;
                    border: 4 px solid grey;
                    border-radius: 45px;
                    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                    transition: all 0.3s ease 0s;
                    cursor: pointer;
                    outline: none;
                    width: 180px; /* Chỉnh sửa chiều rộng của button để cố định kích thước */
                }

                div.stButton > button:hover {
                    background-color: #23c483;
                    box-shadow: 0px 15px 20px rgba(46, 229, 157, 0.4);
                    color: #fff;
                    transform: translateY(-3px);
                }

                div.stButton > button:active {
                    transform: translateY(-1px);
                }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Dùng vòng lặp để tạo các nút button
        for n_row, row in df_display.reset_index().iterrows():
            i = n_row % N_cards_per_row
            if i == 0:
                st.write("---")
                cols = st.columns(N_cards_per_row, gap="large")
            # draw the card
            with cols[n_row % N_cards_per_row]:
                labels = row["Labels"].strip().split()
                for label in labels:
                    # Sử dụng st.markdown để tạo nút button và áp dụng CSS
                    if st.button(label):
                        #print("session stage:" ,st.session_state.stage)
                        #print(label)
                        changeValue("Choose_label", label)
                        changeValue("Click", "1")
                        changeValue("All", "0")
                        changeValue("Search", "0")
                        set_stage(1)
                        st.experimental_rerun()


    elif getValue("Click") == "1":
        #print("Da chọn 1 từ")
        #print(getValue("Choose_label"))
        if st.button("Back", on_click=set_stage(2)):
            print("back lại ban đầu") 
        st.markdown(f'<p style="font-size: 30px; color: black; font-weight: bold;">{getValue("Choose_label")}</p>', unsafe_allow_html=True)
        video_links = df[df['Labels'].str.contains(getValue("Choose_label"))]['Links'].tolist()
        video_id = video_links[0].split('=')[1]
        # Nếu có thêm tham số sau video_id, tiếp tục tách chuỗi bằng '&' và lấy phần tử đầu tiên
        if '&' in video_id:
            video_id = video_id.split('&')[0]
        #print(video_id)
        st.write(f'<iframe width="450" height="350" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)
        if st.session_state.stage == 2:
            changeValue("Choose_label", "0")
            changeValue("Click", "0")
            changeValue("All", "1")
            changeValue("Search", "0")   
            #print("Da thay doi ca gia")
            set_stage(0)

elif app_mode == 'Video Quiz':
    st.title('Video Quiz')
    
    videos = {
        'Video 1': {
            'url': 'https://qipedc.moet.gov.vn/videos/D0006.mp4',
            'correct_answer': 'A'
        },
        'Video 2': {
            'url': 'https://qipedc.moet.gov.vn/videos/D0002.mp4',
            'correct_answer': 'B'
        }
    }

    selected_video = st.selectbox('Chọn video:', list(videos.keys()))

    st.video(videos[selected_video]['url'])

    user_answer = st.radio('What is your answer?', ['A', 'B'])

    if st.button('Check Answer'):
        correct_answer = videos[selected_video]['correct_answer']
        if user_answer == correct_answer:
            st.success('Correct! The answer is {}'.format(correct_answer))
        else:
            st.error('Incorrect. The correct answer is {}'.format(correct_answer))

elif app_mode == 'Action Checker':
    st.title("Action Checker")

    st.markdown("This mode is an extension of the Dictionary mode. It allows users to select a word from the dictionary, record a video of themselves and the system will check if the action of the user matches the selected word.")
    # st.markdown("")
    # Connect to the Google Sheet
    sheet_id = "1dbaXMziDDIQ9Rbt7yoNQPWMSOw72iGs1HNAYwPiu7lU"
    sheet_name = "dictionary"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    @st.cache_data
    def load_data():
        # Read data from Google Sheet
        df = pd.read_csv(url, dtype=str).fillna("")
        return df

    df = load_data()

    if 'stage' not in st.session_state:
        st.session_state.stage = 0
        createCsv()

    def set_stage(stage):
            st.session_state.stage = stage

    def clear_text():
        st.session_state.my_text = st.session_state.widget
        #print("Thay doi text_search")
        changeValue("Choose_label", "0")
        changeValue("Click", "0")
        changeValue("All", "0")
        changeValue("Search", "1")
        st.session_state.widget = ""

    text_search = st.text_input("Enter the word you want to search for in the search bar", value="", 
                key='widget', 
                on_change=clear_text
                ).lower()

    #print("Bat dau lai tu dau")

    text_search = st.session_state.get('my_text', '')


    if getValue("Search") == "0":
        text_search = ""
        
    #print("Text search is:", text_search)

    choose_label = getValue("Choose_label")
    #print("Choose label là:", choose_label)

    if choose_label == "0":
        #print("chay lai")
        print("")

    if choose_label != "0":
        text_search = choose_label
        #print("da chon")

    # Filter the dataframe based on search term
    m1 = df["Labels"].str.contains(text_search)
    df_search = df[m1]

    # Show warning if no matching results found
    if df_search.empty and text_search:
        st.warning("No matching results found for the search term.", icon="⚠️")

    # Show the cards
    N_cards_per_row = 4

    #print("GetAll: ", getValue("All"))

    if not text_search or getValue("All") == "1":
        df_display = df
    else:
        df_display = df_search

    if getValue("Click") == "0":
        st.markdown(
            """
            <style>
                /* Thiết kế lại giao diện cho nút button */
                div.stButton > button {
                    padding: 0.8em 2em;
                    font-size: 8px;
                    text-transform: uppercase;
                    letter-spacing: 2.5px;
                    font-weight: 300;
                    color: #000;
                    background-color: #fff;
                    border: 4 px solid grey;
                    border-radius: 45px;
                    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                    transition: all 0.3s ease 0s;
                    cursor: pointer;
                    outline: none;
                    width: 180px; /* Chỉnh sửa chiều rộng của button để cố định kích thước */
                }

                div.stButton > button:hover {
                    background-color: #23c483;
                    box-shadow: 0px 15px 20px rgba(46, 229, 157, 0.4);
                    color: #fff;
                    transform: translateY(-3px);
                }

                div.stButton > button:active {
                    transform: translateY(-1px);
                }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Dùng vòng lặp để tạo các nút button
        for n_row, row in df_display.reset_index().iterrows():
            i = n_row % N_cards_per_row
            if i == 0:
                st.write("---")
                cols = st.columns(N_cards_per_row, gap="large")
            # draw the card
            with cols[n_row % N_cards_per_row]:
                labels = row["Labels"].strip().split()
                for label in labels:
                    # Sử dụng st.markdown để tạo nút button và áp dụng CSS
                    if st.button(label):
                        changeValue("Choose_label", label)
                        changeValue("Click", "1")
                        changeValue("All", "0")
                        changeValue("Search", "0")
                        set_stage(1)
                        st.rerun()


    elif getValue("Click") == "1":
        if st.button("Back"):
            set_stage(2)
            print("back lại ban đầu") 
        st.markdown(f'<p style="font-size: 30px; color: black; font-weight: bold;">{getValue("Choose_label")}</p>', unsafe_allow_html=True)
        video_links = df[df['Labels'].str.contains(getValue("Choose_label"))]['Links'].tolist()
        video_id = video_links[0].split('=')[1]
        # Nếu có thêm tham số sau video_id, tiếp tục tách chuỗi bằng '&' và lấy phần tử đầu tiên
        if '&' in video_id:
            video_id = video_id.split('&')[0]
        #print(video_id)
        st.write(f'<iframe width="450" height="350" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)

        FRAME_WINDOW = st.image([])

        if 'run' not in st.session_state:
            st.session_state.run = False

        st.button('Record Video', key='record_button', on_click=run, disabled=st.session_state.run)

        if st.session_state.run:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
            record_video(tfile.name, 8, FRAME_WINDOW)

            FRAME_WINDOW.empty()

            st.success('Recording finished!')

            # Convert XVID to H.264 (MP4)
            mp4_filename = tfile.name.replace('.avi', '.mp4')
            subprocess.run(['ffmpeg', '-i', tfile.name, '-vcodec', 'libx264', mp4_filename])

            if os.path.exists(mp4_filename):
                st.video(mp4_filename, format='video/mp4', start_time=0)
                
            status_placeholder = st.empty()
            with status_placeholder:
                st.write('<div style="text-align:center;">Processing...</div>', unsafe_allow_html=True)
            predicted = loading_inference_video(mp4_filename, prediction_fn, ORD2SIGN)
            with status_placeholder:
                status_placeholder.empty()
        
            # Print
            print(predicted)
            st.text_area(label="Prediction", value=predicted, height=50)
            print(getValue("Choose_label"))
            if getValue("Choose_label") == predicted:
                st.success('Correct! Your action matches the selected word.')
            else:
                st.error('Incorrect. Your action does not match the selected word. Please try again.')
            st.session_state.run = False
            st.session_state.stop_recording = False

            if st.button("Rerun", key="rerun"):
                st.rerun()

        if st.session_state.stage == 2:
            changeValue("Choose_label", "0")
            changeValue("Click", "0")
            changeValue("All", "1")
            changeValue("Search", "0")   
            set_stage(0)
            st.rerun()
        