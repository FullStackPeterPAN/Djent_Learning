from keras.models import load_model
from codes import read_audio
import wave
import numpy as np
import matplotlib.pyplot as plt

input_path = "data/train/input/"
read_test_in = read_audio
test_in = read_test_in.read_file(input_path, 0)

model_path = "data/model/model.h5"
model = load_model(model_path)

test_out = model.predict(test_in)

# open a wave file to be written
f = wave.open(r"test_out.wav", "wb")

# set channel, sample width, frame rate
f.setnchannels(1)
f.setsampwidth(read_test_in.get_sample_width())
f.setframerate(read_test_in.get_frame_rate())
# write file
f.writeframes(test_out.tostring())

# plot the wave
time = np.arange(0, read_test_in.get_num_frame()) * (1.0 / read_test_in.get_frame_rate())
plt.plot(time, test_out)
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.title("show wave")
plt.grid('on')
plt.show()

f.close()

