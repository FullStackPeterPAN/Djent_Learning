import os
import errno
import gc
import numpy as np
from numpy import newaxis
from scipy.io import wavfile
import scipy.signal as signal
rate = None
length = None


# file path
expected_path = "data/train/expected/djent_"
input_path = "data/train/input/"
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
    _, _, stft = signal.stft(data, rate, nperseg=npers)
    del data  # clean memory
    stft = stft.T
    return stft


def stft_ri(path):
    stft = get_stft(path)
    real = stft.real
    imag = stft.imag
    del stft  # clean memory
    ri = np.concatenate((real, imag), axis=1)
    del real, imag  # clean memory
    return ri


# transfer to array for training
def array(path, file):
    dim = npers + 2
    # create empty training arrays
    train_in = np.empty([1, 1, dim])
    train_out = np.empty([1, dim])

    # read all audio files to one array
    try:
        # get the last number from file name
        name_num = file.split("_")[-1]

        # read file
        read_in_stft = stft_ri(path)
        read_out_stft = stft_ri(expected_path + str(name_num))

        # shuffle before process
        index = np.arange(len(read_in_stft))
        np.random.shuffle(index)
        read_in_stft = read_in_stft[index]
        read_out_stft = read_out_stft[index]

        # reshape
        read_in = np.array(read_in_stft)
        del read_in_stft   # clean memory
        read_in = read_in[:, newaxis, :]
        if not train_in.size:  # the first array
            train_in = read_in
            train_out = read_out_stft
        else:
            train_in = np.concatenate((train_in, read_in))
            train_out = np.concatenate((train_out, read_out_stft))
            del read_in, read_out_stft  # clean memory

    # catch errors
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
    gc.collect()  # clean memory
    return train_in, train_out
