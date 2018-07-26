import librosa
import numpy as np

# input the audio file
def get_mfcc_data(audio_path):
    y, sr = librosa.load(audio_path)
    mel = librosa.feature.mfcc(y=y, sr=sr)
    mel_data = np.array(mel)
    print(mel_data)
    return mel_data
