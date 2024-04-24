import pandas as pd
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from create_frame_parquet import create_parquet_landmark_from_video, video_capture_loop
from create_frame_parquet import create_output_parquet
import os
import json

# CONFIG
n_class = 250 # Number of classes
ROWS_PER_FRAME = 543 # number of landmarks per frame
UPLOAD_FOLDER = 'AI_recognization/data'  # Folder path
ALLOWED_EXTENSIONS = {'mp4'} # extension for video files
REQUIRED_SIGNATURE = "serving_default"
REQUIRED_OUTPUT = "outputs"

SEL_COLS = pd.read_parquet('resources/data/ASL_Fingerspelling/train_landmarks/1019715464.parquet').columns[1:].tolist()
# print(len(SEL_COLS)) # 1629

xyz = pd.read_parquet('resources/data/ASL_Fingerspelling/train_landmarks/1019715464.parquet')
# print(xyz)

with open('resources/data/ASL_Fingerspelling/inference_args.json', "w") as f:
    json.dump({"selected_columns" : SEL_COLS}, f)

def load_relevant_data_subset(pq_path):
    return pd.read_parquet(pq_path, columns=SEL_COLS)

model_path = "resources/model_weights/ASL_Fingerspelling/model.tflite"

interpreter = tf.lite.Interpreter(model_path)
found_signatures = list(interpreter.get_signature_list().keys())

with open ("resources\data\ASL_Fingerspelling\character_to_prediction_index.json", "r") as f:
    character_map = json.load(f)
rev_character_map = {j:i for i,j in character_map.items()}

# if REQUIRED_SIGNATURE not in found_signatures:
#     raise KernelEvalException('Required input signature not found.')

prediction_fn = interpreter.get_signature_runner("serving_default")

rq_path = 'resources/data/ASL_Fingerspelling/train_landmarks/1019715464.parquet'
frames = load_relevant_data_subset(rq_path)
output = prediction_fn(inputs=frames)
prediction_str = "".join([rev_character_map.get(s, "") for s in np.argmax(output[REQUIRED_OUTPUT], axis=1)])
print(prediction_str)
