import wave
import numpy as np

num_frame = None
num_channel = None
frame_rate = None
num_sample_width = None


# read input file
def read_file(path):
    input_file = wave.open(path, 'rb')
    global num_frame
    num_frame = input_file.getnframes()  # get the number of frames
    global num_channel
    num_channel = input_file.getnchannels()  # get the number of channels
    global frame_rate
    frame_rate = input_file.getframerate()  # get the rate of frames
    global num_sample_width
    num_sample_width = input_file.getsampwidth()  # get the width of sample
    str_data = input_file.readframes(num_frame)  # read all frames
    input_file.close()  # close the file
    wave_data = np.fromstring(str_data, np.int16)  # turn the data to numpy array
    wave_data = np.reshape(wave_data, [num_frame, num_channel])
    return wave_data  # return numpy data


def get_num_channel():
    return num_channel


def get_frame_rate():
    return frame_rate


def get_num_frame():
    return num_frame


def get_sample_width():
    return num_sample_width
