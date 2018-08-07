import wave
import numpy as np

num_frame = None
num_channel = None
frame_rate = None
num_sample_width = None
length = None


# read input file
def read_file(path):
    input_file = wave.open(path, 'rb')
    global num_frame
    num_frame = input_file.getnframes()  # get the number of frames
    global num_channel
    num_channel = input_file.getnchannels()  # get the number of channels, only using mono audio in this program
    global frame_rate
    frame_rate = input_file.getframerate()  # get the rate of frames
    global num_sample_width
    num_sample_width = input_file.getsampwidth()  # get the width of sample
    str_data = input_file.readframes(int(num_frame/frame_rate)*frame_rate)  # read frame_rate*n frames
    input_file.close()  # close the file
    wave_data = np.fromstring(str_data, np.int16)  # turn the data to numpy array
    global length
    length = len(wave_data)
    wave_fft = np.abs(np.fft.fft(wave_data[0: length]))
    wave_data = wave_data.reshape(1, length)
    wave_fft = wave_fft.reshape(1, length)
    data_fft = np.concatenate((wave_data.T, wave_fft.T), axis=1)
    return data_fft


def get_num_channel():
    return num_channel


def get_frame_rate():
    return frame_rate


def get_num_frame():
    return num_frame


def get_length():
    return length


def get_sample_width():
    return num_sample_width
