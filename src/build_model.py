import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def build_model(input_dim: int, units: int = 60):
    model = Sequential()
    model.add(Dense(units, input_dim=input_dim,
              activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid',
              kernel_initializer='glorot_uniform'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model
