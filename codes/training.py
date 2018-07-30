from keras.models import Sequential
from codes import read_audio
from codes import djent_learning

dim = 0
input_path = "data/train/input/"
output_path = "data/train/output/"
expected_path = "data/train/expected/"


# initialize
dl = djent_learning
read_test_in = read_audio
read_test_out = read_audio

# read file
test_in = read_test_in.read_file(input_path, 0)
dim = read_test_in.get_dim()
test_out = read_test_out.read_file(expected_path, 0)

# activate a new model
model = Sequential()

# run the model to evaluate and predict
dl.learning_model(model, test_in, test_out, dim)
