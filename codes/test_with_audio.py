from keras.models import load_model
from codes import read_audio
import numpy as np
import matplotlib.pyplot as plt
import os
import errno
from scipy.io import wavfile

# input path
input_path = "data/test/input/"

for file in os.listdir(input_path):
    try:
        name_num = file.split("_")[-1]

        # get input numpy array
        read_test_in = read_audio
        test_in = read_test_in.get_real_imag(input_path + file)
        test_in = test_in.reshape(read_test_in.get_length(), 1, 2)

        # load the model
        model_path = "data/model/rnn_model.h5"
        weight_path = "data/model/best.hdf5"
        model = load_model(model_path)

        # if exist load weights
        if os.path.exists(weight_path):
            model.load_weights(weight_path)

        # predict the output
        test_out = model.predict(test_in)
        out_fft = test_out[:, 0] + 1j * test_out[:, 1]  # transfer to fft again
        out_data = np.fft.ifft(out_fft).real  # reverse fft
        print(out_data)
        out_data = out_data.astype('int16')  # transfer data to int16

        # write audio
        wavfile.write(r"data/test/output/test_out_" + name_num, read_test_in.get_rate(), out_data)

        # plot the wave
        time = np.arange(0, read_test_in.get_length()) * (1.0 / read_test_in.get_rate())  # time length of the audio
        plt.plot(time, out_data)
        plt.xlabel("Time(s)")
        plt.ylabel("Amplitude")
        plt.title("show wave")
        plt.grid('on')
        plt.show()

    # catch errors
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
