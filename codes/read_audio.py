import numpy as np
from scipy.io import wavfile
import scipy.signal as signal
rate = None
length = None
global npers
npers = 2000


# read input file
def get_data(path):
    global rate
    rate, data = wavfile.read(path)
    global length
    length = int(len(data) / rate) * rate  # adjust the length of data
    data = data[0:length]  # since fft can only read n*frame_rate
    return data


def get_stft(path):
    data = get_data(path)
    f, t, stft = signal.stft(data, rate, nperseg=npers)
    stft = stft.T
    return f, t, stft


def stft_ri(path):
    f, t, stft = get_stft(path)
    real = stft.real
    imag = stft.imag
    ri = np.concatenate((real, imag), axis=1)
    return f, t, ri


def get_rate():
    return rate
