import wave
import struct
import numpy as np
np.set_printoptions(suppress=False)
import matplotlib.pyplot as plt

dim = 0
input_path = "data/train/input/"
output_path = "data/train/output/"
expected_path = "data/train/expected/"


# read input file
def read_file(path, i):
    input_filepath = path

    # choose file name
    if path == input_path:
        input_filename = "clean_" + str(i) + ".wav"
    else:
        input_filename = "tokyo_drive_" + str(i) + ".wav"

    input_file = wave.open(input_filepath + input_filename, 'rb')

    global num_frame
    num_frame = input_file.getnframes()  # get the number of frames
    num_channel = input_file.getnchannels()  # get the number of channels

    global frame_rate
    frame_rate = input_file.getframerate()  # get the rate of frames

    global num_sample_width
    num_sample_width = input_file.getsampwidth()  # get the width of sample

    str_data = input_file.readframes(num_frame)  # read all frames

    input_file.close()  # close the file
    wave_data = np.fromstring(str_data, np.int16)  # turn the data to numpy array
    print(wave_data)
    np.savetxt("wave_data", wave_data)

    # plot the wave
    time = np.arange(0, num_frame) * (1.0 / frame_rate)
    plt.plot(time, wave_data)
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("show wave")
    plt.grid('on')
    plt.show()

    return wave_data  # return numpy data


def get_frame_rate():
    return frame_rate


def get_num_frame():
    return num_frame


def get_sample_width():
    return num_sample_width

