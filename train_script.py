# import librosa
# import os
# import librosa.display
from src.model import Model
from config.config import HYPARMS
from DataConverter import inputdata
from FTP_Manager.downloadfromftp import downloadfromftp


downloadfromftp()
model = Model()
data_sets = inputdata.read_sound_datasets(HYPARMS.train_data_dir,
                                          HYPARMS.test_data_dir,
                                          reshape=False)

model.load_data(data_sets.train, data_sets.test)
model.train()
