from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, SimpleRNN, Activation
from keras.optimizers import Adadelta
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
model.add(SimpleRNN(units=2, input_shape=(1, 2), activation=None, return_sequences=True, return_state=False))
model.add(LeakyReLU(alpha=0.01))
model.add(Dropout(0.3))
model.add(SimpleRNN(units=2, activation=None, return_sequences=False, return_state=False))
model.add(LeakyReLU(alpha=0.01))
model.add(Dropout(0.3))
model.add(Dense(units=2))


optimizer = Adadelta()
model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

# create empty training arrays
train_in = np.empty([1, 1, 2])
train_out = np.empty([1, 2])

# read all audio files to one array
for file in os.listdir(input_path):
    try:
        # get the last number from file name
        name_num = file.split("_")[-1]

        # initialize
        read_train_in = read_audio
        read_train_out = read_audio

        # read file
        read_in = read_train_in.get_real_imag(input_path + file)
        read_out = read_train_out.get_real_imag(expected_path + name_num)
        read_in = read_in.reshape(read_train_in.get_length(), 1, 2)
        if not train_in.size:  # the first array
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
model.fit(train_in, train_out, epochs=4, batch_size=44100)  # test with only 1 epoch

# evaluate the model
loss, accuracy = model.evaluate(train_in, train_out)
print(loss, accuracy)

# save the model
model.save("data/model/model.h5")
