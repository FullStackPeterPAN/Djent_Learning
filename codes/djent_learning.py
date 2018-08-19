from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, LSTM
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from codes import read_audio
import numpy as np
from numpy import newaxis
import os
import errno

# file path
expected_path = "data/train/expected/shorten_drive_"
input_path = "data/train/input/"
model_path = "data/model/rnn_drive_model.h5"
weight_path = "data/model/weights_drive.best.hdf5"

# activate a new model
model = Sequential()

# input shape needs to be changed to (2, ) if using stereo audio
model.add(LSTM(512, return_sequences=True, input_shape=(1, 442)))  # nperseg of stft + 2
model.add(LeakyReLU())
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=True))
model.add(LeakyReLU())
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(LeakyReLU())
model.add(Dropout(0.2))
model.add(Dense(442))  # nperseg of stft + 2


''' Model 1
model.add(SimpleRNN(units=2, input_shape=(1, 2), activation='tanh', return_sequences=True, return_state=False))
model.add(Dropout(0.1))
model.add(SimpleRNN(units=2, activation='tanh', return_sequences=True, return_state=False))  # able to add more layers
model.add(Dropout(0.1))  # by repeating these two lines
model.add(SimpleRNN(units=2, activation='tanh', return_sequences=False, return_state=False))
model.add(Dropout(0.1))
model.add(Dense(units=2))
'''

'''
# control the size of fit generator
def data_generator(data, targets, batch_size):
    batches = len(data) + batch_size - 1
    while(True):
        for i in range(batches):
            x = data[i * batch_size: (i + 1) * batch_size]
            y = targets[i * batch_size: (i + 1) * batch_size]
            yield (x, y)
'''


# training method
def array(path):
    # create empty training arrays
    train_in = np.empty([1, 1, 442])  # nperseg of stft + 2
    train_out = np.empty([1, 442])  # nperseg of stft + 2

    # read all audio files to one array
    for file in os.listdir(path):
        try:
            # get the last number from file name
            name_num = file.split("_")[-1]

            # initialize
            read_train_in = read_audio
            read_train_out = read_audio

            # read file
            read_in_f, read_in_t, read_in_stft = read_train_in.stft_ri(path + file)
            read_out_f, read_out_t, read_out_stft = read_train_out.stft_ri(expected_path + name_num)
            read_in = np.array(read_in_stft)
            read_in = read_in[:, newaxis, :]
            if not train_in.size:  # the first array
                train_in = read_in
                train_out = read_out_stft
            else:
                train_in = np.concatenate((train_in, read_in))
                train_out = np.concatenate((train_out, read_out_stft))

        # catch errors
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise
    return train_in, train_out

# count folder numbers
count = 0
for folder in os.listdir(input_path):
    count = count + 1

# processing
for j in range(3, count+1):

    f = input_path + "clean" + str(j) + "/"  # folder path
    train_x, train_y = array(f)

    # if exist load weights
    if os.path.exists(weight_path):
        model.load_weights(weight_path)

    # compile model
    optimizer = Adadelta()
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    # add check point
    checkpoint = ModelCheckpoint(weight_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callback_list = [checkpoint]  # only save the best model

    # fit the model
    model.fit(x=train_x, y=train_y, validation_split=0.33,
              batch_size=500, epochs=8, callbacks=callback_list, verbose=1)

    # evaluate the model
    loss, accuracy = model.evaluate(train_x, train_y, verbose=1)
    print(loss, accuracy)

    # save the model
    model.save(model_path)
