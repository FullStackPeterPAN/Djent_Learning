import wave
from keras.utils import to_categorical
from keras.layers.core import Dense, Dropout, Activation
import scipy.io as sio
import numpy as np
import os


def learning_model(model, input_data, expected_data):
    model.add(Dense(128, input_dim=1, activation='relu'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['mean_squared_error'])

    train_in = input_data
    train_out = expected_data

    print(train_in)
    print(train_out)

    # train the model
    model.fit(train_in, train_out, epochs=100, batch_size=1024)

    # save the model
    model.save("data/model/model.h5")

    # evaluate the model
    loss, accuracy = model.evaluate(train_in, train_out)
    print(loss, accuracy)
