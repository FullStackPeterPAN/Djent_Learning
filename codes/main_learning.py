# import necessary stuff

import wave
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import scipy.io as sio
import numpy as np
from codes import read_audio
import os

dim = 0
input_path = "F:/WorkSpace/DL_Sound_Effect/codes/data/train/input/"
output_path = "F:/WorkSpace/DL_Sound_Effect/codes/data/train/output/"
expected_path = "F:/WorkSpace/DL_Sound_Effect/codes/data/train/expected/"


def learning_model(input_data, expected_data):
    # a sequential model for testing
    model = Sequential()
    model.add(Dense(1, input_dim=dim, activation='relu'))
    model.add(Dense(dim, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])   # compile the model

    train_in = input_data
    train_out = expected_data.T

    # train the model
    model.fit(train_in, train_out, epochs=1, batch_size=1)

    # save the model
    model.save("F:/WorkSpace/DL_Sound_Effect/codes/data/model/model.h5")

    # evaluate the model
    loss, accuracy = model.evaluate(train_in, train_out)
    print(loss, accuracy)


ra = read_audio
test_in = ra.read_file(input_path, 0)
dim = ra.get_dim()
test_out = ra.read_file(expected_path, 0)
# run the model to evaluate and predict
learning_model(test_in, test_out)
