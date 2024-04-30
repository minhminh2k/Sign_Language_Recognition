import pandas as pd
import numpy as np
import tensorflow as tf
from create_frame_parquet import create_parquet_landmark_from_video, video_capture_loop
from create_frame_parquet import create_output_parquet

import os
import cv2
import mediapipe as mp

# CONFIG
N_CLASSES = 250 # Number of classes
ROWS_PER_FRAME = 543 # number of landmarks per frame
FILE_EXTENSIONS = {'mp4'} # extension for video files


def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def loading_model():
    interpreter = tf.lite.Interpreter(model_path="resources/model_weights/ISLR/model.tflite")
    found_signatures = list(interpreter.get_signature_list().keys())

    # if REQUIRED_SIGNATURE not in found_signatures:
    #     raise KernelEvalException('Required input signature not found.')

    prediction_fn = interpreter.get_signature_runner("serving_default")
    return prediction_fn

def loading_class():
    train = pd.read_csv('resources/data/ISLR/train.csv')
    train['sign_ord'] = train['sign'].astype('category').cat.codes

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
    
def inference_ASL(video_path="videos/hello.mp4"):
    interpreter = tf.lite.Interpreter(model_path="resources/model_weights/ISLR/model.tflite")
    found_signatures = list(interpreter.get_signature_list().keys())

    # if REQUIRED_SIGNATURE not in found_signatures:
    #     raise KernelEvalException('Required input signature not found.')

    prediction_fn = interpreter.get_signature_runner("serving_default")

    train = pd.read_csv('resources/data/ISLR/train.csv')
    train['sign_ord'] = train['sign'].astype('category').cat.codes

    SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
    ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

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

    # print(SIGN2ORD) # Dictionaries with 250 classes and id
    # print(ORD2SIGN)

if __name__ == "__main__":
    # print(inference_ASL())
    print(loading_inference_video(video_path="videos/hello.mp4", prediction_fn = loading_model(), ORD2SIGN = loading_class()))