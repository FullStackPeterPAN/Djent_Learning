from keras.models import Sequential
from codes import read_audio
from codes import djent_learning
import os
import errno

expected_path = "data/train/expected/tokyo_drive_"
input_path = "data/train/input/"

# activate a new model
model = Sequential()

for file in os.listdir(input_path):
    try:
        name_num = file.split("_")[-1]
        # initialize
        dl = djent_learning
        read_train_in = read_audio
        read_train_out = read_audio

        # read file
        train_in = read_train_in.read_file(input_path + file)
        train_out = read_train_out.read_file(expected_path + name_num)

        # run the model to evaluate and predict
        dl.learning_model(model, train_in, train_out, read_train_in.get_num_channel())

    # catch errors
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
