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
        test_in = read_test_in.get_data_fft(input_path + file)
        test_in = test_in.reshape(read_test_in.get_length(), 1, 2)

        # load the model
        model_path = "data/model/model.h5"
        model = load_model(model_path)

        # predict the output
        test_out = model.predict(test_in)
        test_out = np.fft.ifft(test_out)
        print(test_out)
        np.savetxt(r'b', test_out)

        # open a wave file to be written
        wavfile.write(r"data/test/output/test_out_" + name_num, read_test_in.get_rate(), test_out)

        # plot the wave
        time = np.arange(0, read_test_in.get_length()) * (1.0 / read_test_in.get_rate())
        plt.figure()
        for i in range(0, 1):
            plt.subplot(1, 1, i + 1)
            plt.plot(time, test_out[:, i])
            plt.xlabel("Time(s)")
            plt.ylabel("Amplitude")
            plt.title("show wave")
            plt.grid('on')
            plt.show()

    # catch errors
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
