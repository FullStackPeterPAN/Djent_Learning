from codes import read_audio
import numpy as np

r = read_audio
input_path = "data/train/input/"
test_in = r.read_file(input_path, 0)
np.savetxt("x", test_in)

