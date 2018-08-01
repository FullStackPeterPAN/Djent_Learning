from keras.models import Sequential
from codes import read_audio
from codes import djent_learning

input_path = "data/train/input/clean_0.wav"
expected_path = "data/train/expected/tokyo_drive_0.wav"

# initialize
dl = djent_learning
read_train_in = read_audio
read_train_out = read_audio

# read file
train_in = read_train_in.read_file(input_path)
train_out = read_train_out.read_file(expected_path)

# activate a new model
model = Sequential()

# run the model to evaluate and predict
dl.learning_model(model, train_in, train_out)
