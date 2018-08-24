from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, LSTM, SimpleRNN
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from codes import read_audio as ra
from codes.read_audio import npers
import numpy as np
import os
import gc


dim = npers + 2  # nperseg of stft + 2

# file path
expected_path = "data/train/expected/djent_"
input_path = "data/train/input/"
model_path = "data/model/lstm_djent_model.h5"
weight_path = "data/model/weights_djent.best.hdf5"

# activate a new model
model = Sequential()

# input shape needs to be changed to (2, ) if using stereo audio
model.add(LSTM(dim, return_sequences=True, input_shape=(1, dim)))
model.add(LeakyReLU())
model.add(Dropout(0.2))
model.add(LSTM(dim, return_sequences=True))
model.add(LeakyReLU())
model.add(Dropout(0.2))
model.add(LSTM(dim, return_sequences=False))
model.add(LeakyReLU())
model.add(Dropout(0.2))
model.add(Dense(dim))

''' trying with RNN model
model.add(SimpleRNN(units=dim, input_shape=(1, dim), activation='relu', return_sequences=True, return_state=False))
model.add(Dropout(0.1))
model.add(SimpleRNN(units=dim, activation='relu', return_sequences=True, return_state=False))  # able to add more layers
model.add(Dropout(0.1))  # by repeating these two lines
model.add(SimpleRNN(units=dim, activation='relu', return_sequences=False, return_state=False))
model.add(Dropout(0.1))
model.add(Dense(units=dim))
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
for epoch in range(0, 100):  # adjust the size of epochs

    # training method
    for i in range(0, 12):  # 0 can be changed to the file wanted to be started

        # mix files for reducing nan
        # read one different file each time
        train_x, train_y = ra.array(i)
        for j in range(1, 3):  # concatenate separately to avoid memory error
            n = i + j
            if (i+j) > 13:  # for the last few files
                n = n - 14
            x, y = ra.array(n)
            train_x = np.concatenate((train_x, x))
            train_y = np.concatenate((train_y, y))
            del x, y  # clean memory

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
        model.fit(x=train_x, y=train_y, validation_split=0.05, batch_size=1000, epochs=10,
                  callbacks=callback_list, verbose=1, shuffle=True)

        # evaluate the model
        loss, accuracy = model.evaluate(train_x, train_y, verbose=1)
        print(loss, accuracy)

        # save the model
        if loss != 'nan':
            model.save(model_path)

        # clean memory for next file
        del train_x, train_y, loss, accuracy
        gc.collect()
