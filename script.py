# import librosa
# import os
# import librosa.display
from src.model import Model
from config.config import HYPARMS
from DataConverter import inputdata



model = Model()
data_sets = inputdata.read_sound_datasets(HYPARMS.train_data_dir,
                                          HYPARMS.test_data_dir,
                                          reshape=False)

model.load_data(data_sets.train, data_sets.test)
model.train()
#
# dirpath = "data"
# filepath = "하1.m4a"
# path = os.path.join(dirpath, filepath)
# y, sr = librosa.core.load(path) # sr = sampling rate
#
# mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
# print(mfcc.shape)
#
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,4))
# librosa.display.specshow(mfcc, x_axis='time')
# plt.colorbar()
# plt.title('MFCC')
# plt.tight_layout()
# plt.show()