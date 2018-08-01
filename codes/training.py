from keras.models import Sequential
from codes import read_audio
from codes import djent_learning

input_path = "data/train/input/"
output_path = "data/train/output/"
expected_path = "data/train/expected/"

# initialize
dl = djent_learning
read_train_in = read_audio
read_train_out = read_audio

# read file
train_in = read_train_in.read_file(input_path, 0)
train_out = read_train_out.read_file(expected_path, 0)

# activate a new model
model = Sequential()

# run the model to evaluate and predict
dl.learning_model(model, train_in, train_out)
