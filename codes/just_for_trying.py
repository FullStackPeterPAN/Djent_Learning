from codes import read_audio
import numpy as np

input_path = "data/train/input/"
expected_path = "data/train/expected/"

read_test_in = read_audio
read_test_out = read_audio
# read file
test_in = read_test_in.read_file(input_path, 0)