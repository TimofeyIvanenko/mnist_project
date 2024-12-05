# model.py

import tensorflow.keras as keras
from tensorflow.keras.layers import Flatten, Dense, Dropout

def build_model(input_shape=(28, 28)):
    model = keras.Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    return model
