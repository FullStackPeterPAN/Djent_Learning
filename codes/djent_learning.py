from keras.models import Sequential, load_model
from keras.layers.advanced_activations import LeakyReLU  # reduce zero output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from codes import read_audio
import numpy as np
import os
import errno

expected_path = "data/train/expected/tokyo_drive_"
input_path = "data/train/input/"
model_path = "data/model/model.h5"

# activate a new model
model = Sequential()

# input shape needs to be changed to 2 if using stereo audio
model.add(LeakyReLU(alpha=0.01, input_shape=(1, 3)))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(Dropout(0.5))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# create empty training arrays
train_in = np.empty([1, 1, 3])
train_out = np.empty([1, 2])

for file in os.listdir(input_path):
    try:
        # get the last number from file name
        name_num = file.split("_")[-1]

        # initialize
        read_train_in = read_audio
        read_train_out = read_audio

        # read file
        read_in = read_train_in.get_data_fft(input_path + file)
        read_out = read_train_out.get_real_imag(expected_path + name_num)
        read_in = read_in.reshape(read_train_in.get_length(), 1, 3)
        if not train_in.size:
            train_in = read_in
            train_out = read_out
        else:
            train_in = np.concatenate((train_in, read_in))
            train_out = np.concatenate((train_out, read_out))

    # catch errors
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

# train the model
model.fit(train_in, train_out, epochs=1, batch_size=300000)  # test with only 1 epoch

# evaluate the model
loss, accuracy = model.evaluate(train_in, train_out)
print(loss, accuracy)

# save the model
model.save("data/model/model.h5")
