import wave
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import scipy.io as sio
import numpy as np
from codes import read_audio
import os
from codes import djent_learning

dim = 0
input_path = "F:/WorkSpace/Djent_Learning/codes/data/train/input/"
output_path = "F:/WorkSpace/Djent_Learning/codes/data/train/output/"
expected_path = "F:/WorkSpace/Djent_Learning/codes/data/train/expected/"

ra = read_audio
dl = djent_learning
test_in = ra.read_file(input_path, 0)
dim = ra.get_dim()
test_out = ra.read_file(expected_path, 0)
# run the model to evaluate and predict
dl.learning_model(test_in, test_out, dim)
