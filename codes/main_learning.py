# import necessary stuff

import wave
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import scipy.io as sio
import numpy as np
import os


dim = 0
input_path = "F:/WorkSpace/DL_Sound_Effect/codes/data/train/input/"
output_path = "F:/WorkSpace/DL_Sound_Effect/codes/data/train/output/"
expected_path = "F:/WorkSpace/DL_Sound_Effect/codes/data/train/expected/"


# read input file
def read_file(path, i):
    input_filepath = path
    input_filename = None
    if path == input_path:
        input_filename = "clean_" + str(i) + ".wav"
    else:
        input_filename = "tokyo_drive_" + str(i) + ".wav"
    input_file = wave.open(input_filepath + input_filename, 'rb')
    num_frame = input_file.getnframes()  # get the number of frames

    # count frames number if it is clean audio
    global frames
    frames = num_frame
    num_channel = input_file.getnchannels()  # get the number of channels
    frame_rate = input_file.getframerate()  # get the rate of frames
    num_sample_width = input_file.getsampwidth()  # get the width of sample
    str_data = input_file.readframes(num_frame)  # read all frames
    input_file.close()  # close the file
    wave_data = np.fromstring(str_data, dtype=np.short)  # turn the data to numpy array
    wave_data.shape = -1, num_channel  # shape the data depending on the number of channels
    wave_data = wave_data.T  # turn the wave data
    wave_data = wave_data
    global dim
    dim = len(wave_data[0])
    return wave_data  # return numpy data


def learning_model(input_data, expected_data):
    # a sequential model for testing
    model = Sequential()
    model.add(Dense(1, input_dim=dim, activation='relu'))
    model.add(Dense(dim, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])   # compile the model

    train_in = input_data
    train_out = expected_data

    # train the model
    model.fit(train_in, train_out, epochs=1, batch_size=1)

    # evaluate the model
    loss, accuracy = model.evaluate(train_in, train_out)
    print(loss, accuracy)


test_in = read_file(input_path, 0)
test_out = read_file(expected_path, 0)
# run the model to evaluate and predict
learning_model(test_in, test_out)
