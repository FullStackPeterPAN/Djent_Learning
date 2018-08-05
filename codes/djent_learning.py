from keras.models import Sequential, load_model
from keras.layers.advanced_activations import LeakyReLU  # reduce zero output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from codes import read_audio
import os
import errno

expected_path = "data/train/expected/tokyo_drive_"
input_path = "data/train/input/"
model_path = "data/model/model.h5"

for file in os.listdir(input_path):
    try:
        # get the last number from file name
        name_num = file.split("_")[-1]

        # initialize
        read_train_in = read_audio
        read_train_out = read_audio

        # read file
        train_in = read_train_in.read_file(input_path + file)
        train_out = read_train_out.read_file(expected_path + name_num)
        train_in = train_in.reshape(read_train_in.get_num_frame(),1, 1)

        if os.path.isfile(model_path):

            # load existed model
            model = load_model(model_path)
        else:

            # activate a new model
            model = Sequential()

            # input shape needs to be changed to 2 if using stereo audio
            model.add(LSTM(64, return_sequences=True, input_shape=(1, 1), activation=LeakyReLU(alpha=0.3)))
            model.add(Dropout(0.2))
            model.add(LSTM(64, return_sequences=False, activation='tanh'))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            model.add(Activation('softmax'))

            # compile the model
            model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        # train the model
        model.fit(train_in, train_out, epochs=2, batch_size=200000)

        # evaluate the model
        loss, accuracy = model.evaluate(train_in, train_out)
        print(loss, accuracy)

        # save the model
        model.save("data/model/model.h5")

    # catch errors
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise


