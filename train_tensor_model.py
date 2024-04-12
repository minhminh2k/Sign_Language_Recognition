import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou', 'fuckyou'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

label_map = {label:num for num, label in enumerate(actions)}

def main():
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    print('Sequences: ', np.array(sequences).shape)
    print('Label: ', np.array(labels).shape)
    
    # Divide datasets 
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    print('Test:', y_test.shape)
    
    # Logger
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    
    # Create Tesorflow model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) #1662
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    # Adjust epochs for training
    model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback]) 
    # model.summary()
    
    
    # Predict 
    res = [.2, 0.6, 0.1, 0.1]
    print("Test input and output: ", actions[np.argmax(res)])
    
    res = model.predict(X_test)
    print("Label: ", actions[np.argmax(res[4])])
    print("Predict: ", actions[np.argmax(y_test[4])])
    
    # Save weights
    model.save('action.h5')
    del model
    
    # Load weights for testing
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258))) #1662
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    
    model.load_weights('action.h5')
    
if __name__ == '__main__':
    main()