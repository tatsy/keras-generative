import h5py
import numpy as np

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
    dset.images = np.asarray(f['images'], 'float32') / 255.0
    dset.attribs = np.asarray(f['labels'], 'float32')
    dset.names = np.asarray(f['label_names'])

    return dset
