import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

def create_model(actions):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) # 1662 if using face contours
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.load_weights('action.h5')
    return model

actions = np.array(['hello', 'thanks', 'iloveyou', 'fuckyou'])

model_tensor = create_model(actions)
