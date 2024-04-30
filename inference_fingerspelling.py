import pandas as pd
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from create_parquet_fingerspelling import create_output_parquet_fingerspelling

import os
import json

# CONFIG
FILE_EXTENSIONS = {'mp4'} # extension for video files
REQUIRED_SIGNATURE = "serving_default"
REQUIRED_OUTPUT = "outputs"

with open('resources/data/ASL_Fingerspelling/inference_args.json', 'r') as file:
    data = json.load(file)

SEL_COLS = data['selected_columns']
# print(len(SEL_COLS)) # 1629

# Write to the Json file
# with open('resources/data/ASL_Fingerspelling/inference_args.json', "w") as f:
#     json.dump({"selected_columns" : SEL_COLS}, f)

def load_relevant_data_subset_fingerspelling(pq_path):
    return pd.read_parquet(pq_path, columns=SEL_COLS)

def loading_model_fingerspelling():
    model_path = "resources/model_weights/ASL_Fingerspelling/model.tflite"

    interpreter = tf.lite.Interpreter(model_path)
    found_signatures = list(interpreter.get_signature_list().keys())

    # if REQUIRED_SIGNATURE not in found_signatures:
    #     raise KernelEvalException('Required input signature not found.')

    prediction_fn = interpreter.get_signature_runner("serving_default")
    
    return prediction_fn

def loading_charater_fingerspelling():
    with open ("resources/data/ASL_Fingerspelling/character_to_prediction_index.json", "r") as f:
        character_map = json.load(f)
    rev_character_map = {j:i for i,j in character_map.items()}
    return rev_character_map

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
        # print("Video is invalid!!!")
        return e


if __name__ == "__main__":
    print(loading_inference_video_fingerspelling(video_path="videos/fingerspelling/123all451.mp4", 
                                                 prediction_fn=loading_model_fingerspelling(), 
                                                 rev_character_map=loading_charater_fingerspelling()))