from keras.models import load_model
from codes import read_audio
import wave
import numpy as np
import matplotlib.pyplot as plt
import os
import errno

# input path
input_path = "data/test/input/"

for file in os.listdir(input_path):
    try:
        name_num = file.split("_")[-1]

        # get input numpy array
        read_test_in = read_audio
        test_in = read_test_in.read_file(input_path + file)
        test_in = test_in.reshape(read_test_in.get_num_frame(), 1, 1)

        # load the model
        model_path = "data/model/model.h5"
        model = load_model(model_path)

        # predict the output
        test_out = model.predict(test_in)
        print(test_out.tostring())
        # open a wave file to be written
        f = wave.open(r"data/test/output/test_out_" + name_num, "wb")

        # set channel, sample width, frame rate
        f.setnchannels(read_test_in.get_num_channel())
        f.setsampwidth(read_test_in.get_sample_width())
        f.setframerate(read_test_in.get_frame_rate())

        # write file
        f.writeframes(test_out.tostring())

        # plot the wave
        time = np.arange(0, read_test_in.get_num_frame()) * (1.0 / read_test_in.get_frame_rate())
        plt.figure()
        for i in range(0, read_test_in.get_num_channel()):
            plt.subplot(read_test_in.get_num_channel(), 1, i + 1)
            plt.plot(time, test_out[:, i])
            plt.xlabel("Time(s)")
            plt.ylabel("Amplitude")
            plt.title("show wave")
            plt.grid('on')
            plt.show()

        f.close()

    # catch errors
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
