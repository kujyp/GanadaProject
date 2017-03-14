import librosa
import os
import librosa.display


dirpath = "data"
filepath = "하1.m4a"
path = os.path.join(dirpath, filepath)
y, sr = librosa.core.load(path) # sr = sampling rate

mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
print(mfcc.shape)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()