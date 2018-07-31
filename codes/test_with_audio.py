from keras.models import load_model
from codes import read_audio
import wave
import numpy as np

input_path = "data/train/input/"
read_test_in = read_audio
test_in = read_test_in.read_file(input_path, 0)
print(test_in)

model_path = "data/model/model.h5"
model = load_model(model_path)
test_out = model.predict(test_in)
print(test_out)
wave_data = test_out.reshape(1, read_test_in.get_dim())

wave_data = wave_data.astype(np.short)
print(wave_data)

# 打开WAV文档
f = wave.open(r"test_out.wav", "wb")

# 配置声道数、量化位数和取样频率
f.setnchannels(1)
f.setsampwidth(read_test_in.get_sample_width())
f.setframerate(read_test_in.get_frame_rate())
# 将wav_data转换为二进制数据写入文件
f.writeframes(wave_data.tostring())
f.close()

