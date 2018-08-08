import wave
import numpy as np
from scipy.io import wavfile

rate = None
length = None


# read input file
def get_data(path):
    global rate
    rate, data = wavfile.read(path)
    global length
    length = int(len(data) / rate) * rate  # adjust the length of data
    data = data[0:length]  # since fft can only read n*frame_rate
    return data


def get_fft(path):
    data = get_data(path)
    fft = (np.fft.fft(data))
    return fft


def get_real_imag(path):
    fft = get_fft(path)
    real = fft.real.reshape(1, get_length())  # reshape the real part of fft
    imag = fft.imag.reshape(1, get_length())  # reshape the imaginary part of fft
    real_imag = np.concatenate((real.T, imag.T), axis=1)
    return real_imag


def get_data_fft(path):
    data = get_data(path)
    data = data.reshape(1, get_length())  # reshape the data array
    fft = get_fft(path)
    real = fft.real.reshape(1, get_length())  # reshape the real part of fft
    imag = fft.imag.reshape(1, get_length())  # reshape the imaginary part of fft
    data_fft = np.concatenate((data.T, real.T, imag.T), axis=1)
    return data_fft


def get_rate():
    return rate


def get_length():
    return length
