# import necessary stuff

import wave
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import os

filepath = "F:/WorkSpace/DL_Sound_Effect/codes/data/train/"


# read input file
def read_file(path):
    input_filepath = path + "input/"
    input_filenames = os.listdir(input_filepath)
    for input_filename in input_filenames:
        input_file = wave.open(input_filepath+input_filename, 'rb')
        num_frame = input_file.getnframes()  # get the number of frames
        num_channel = input_file.getnchannels()  # get the number of channels
        frame_rate = input_file.getframerate()  # get the rate of frames
        num_sample_width=input_file.getsampwidth()  # get the width of sample
        str_data = input_file.readframes(num_frame)  # read all frames
        input_file.close()  # close the file
        wave_data = np.fromstring(str_data, dtype=np.short)  # turn the data to numpy array
        wave_data.shape = -1, num_channel  # shape the data depending on the number of channels
        wave_data = wave_data.T  # turn the
        wave_data = wave_data
        return wave_data, frame_rate


print(read_file(filepath))

