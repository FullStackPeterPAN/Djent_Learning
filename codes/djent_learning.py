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
model.add(LeakyReLU(alpha=0.01, input_shape=(1, 1)))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(Dropout(0.5))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# training arrays
train_in = np.empty([1, 1, 1])
train_out= np.empty([1])

for file in os.listdir(input_path):
    try:
        # get the last number from file name
        name_num = file.split("_")[-1]

        # initialize
        read_train_in = read_audio
        read_train_out = read_audio

        # read file
        read_in = read_train_in.read_file(input_path + file)
        read_out = read_train_out.read_file(expected_path + name_num)
        read_in = read_in.reshape(read_train_in.get_num_frame(), 1, 1)
        if not train_in.size:
            train_in = read_in
            train_out = read_out
        else:
            train_in = np.concatenate((train_in, read_in))
            train_out = np.concatenate((train_out, read_out))
        print(train_out)
        print(train_in)

    # catch errors
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

# train the model
model.fit(train_in, train_out, epochs=5, batch_size=200000)

# evaluate the model
loss, accuracy = model.evaluate(train_in, train_out)
print(loss, accuracy)

# save the model
model.save("data/model/model.h5")
