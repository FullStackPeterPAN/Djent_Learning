# import necessary stuff

import wave
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.models import load_model
import scipy.io as sio
import numpy as np
import os

model_path = "F:/WorkSpace/Djent_Learning/codes/data/model/model.h5"
model = load_model(model_path)
