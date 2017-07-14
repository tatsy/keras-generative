import h5py
import numpy as np

class Dataset(object):
    def __init__(self):
        self.images = None
        self.attribs = None
        self.names = None

    def __len__(self):
        return len(self.images)

def load_data(filename, size=-1):
    f = h5py.File(filename)

    dset = Dataset()
    dset.images = np.asarray(f['images'], 'float32') / 255.0
    dset.attrs = np.asarray(f['attrs'], 'float32')
    dset.attr_names = np.asarray(f['attr_names'])

    if size > 0:
        dset.images = dset.images[:size]
        dset.attrs = dset.attrs[:size]

    return dset
