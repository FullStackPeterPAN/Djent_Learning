from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from codes import read_audio
import os
import errno

expected_path = "data/train/expected/tokyo_drive_"
input_path = "data/train/input/"

# activate a new model
model = Sequential()

model.add(Dense(256, input_shape=(1,)))  # input shape needs to be changed to 2 if using stereo audio
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))  # dense needs to be changed to 2 if using stereo audio
model.add(Activation('softmax'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

for file in os.listdir(input_path):
    try:
        name_num = file.split("_")[-1]

        # initialize
        read_train_in = read_audio
        read_train_out = read_audio

        # read file
        train_in = read_train_in.read_file(input_path + file)
        train_out = read_train_out.read_file(expected_path + name_num)

        # train the model
        model.fit(train_in, train_out, epochs=5, batch_size=10000)

        # evaluate the model
        loss, accuracy = model.evaluate(train_in, train_out)
        print(loss, accuracy)

    # catch errors
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

# save the model
model.save("data/model/model.h5")
