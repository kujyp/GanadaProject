from DataConverter.inputdata import input_snd_mfcc_data
from DataConverter.load.snd_loader import load_allsnd_data
import IPython.display
from DataConverter.visualize.visualizer_numpy_sound import plot_sound
import numpy as np


snd_mfcc, labels = input_snd_mfcc_data("data")
# ys, srs = load_allsnd_data("data/ga")
#
# idx = 2
# y = ys[idx]
# sr = srs[idx]

# plot_sound(y_crop,sr)
# plot_sound(y, sr)
