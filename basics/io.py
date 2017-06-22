import h5py
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def load_celebA(filename):
    f = h5py.File(filename)
    return np.asarray(f['images'], np.float32)
