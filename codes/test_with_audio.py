from keras.models import load_model
from codes import read_audio
import wave
import numpy as np
import matplotlib.pyplot as plt

input_path = "data/train/input/"
read_test_in = read_audio
test_in = read_test_in.read_file(input_path, 0)

np.savetxt("y", test_in)

model_path = "data/model/model.h5"
model = load_model(model_path)
print(model.summary())

test_out = model.predict(test_in)
wave_data = test_out.astype(np.int16)

np.savetxt("z", wave_data)
# 打开WAV文档
f = wave.open(r"test_out.wav", "wb")

# 配置声道数、量化位数和取样频率
f.setnchannels(1)
f.setsampwidth(read_test_in.get_sample_width())
f.setframerate(read_test_in.get_frame_rate())
# 将wav_data转换为二进制数据写入文件
f.writeframes(wave_data.tostring())

# plot the wave
time = np.arange(0, read_test_in.get_num_frame()) * (1.0 / read_test_in.get_frame_rate())
plt.plot(time, wave_data)
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.title("show wave")
plt.grid('on')
plt.show()

f.close()

