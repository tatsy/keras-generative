import h5py
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Dataset(object):
    def __init__(self):
        self.images = None
        self.attribs = None
        self.names = None

    def __len__(self):
        return len(self.images)

def load_celebA(filename):
    f = h5py.File(filename)

    dset = Dataset()
    dset.images = np.asarray(f['images'], 'float32')
    dset.attribs = np.asarray(f['labels'], 'uint8')
    dset.names = np.asarray(f['label_names'])

    return dset
