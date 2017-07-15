import h5py
import numpy as np

class Dataset(object):
    def __init__(self):
        self.images = None

    def __len__(self):
        return len(self.images)

class ConditionalDataset(Dataset):
    def __init__(self):
        super(ConditionalDataset, self).__init__()
        self.attrs = None
        self.attr_names = None

def load_data(filename, size=-1):
    f = h5py.File(filename)

    dset = ConditionalDataset()
    dset.images = np.asarray(f['images'], 'float32') / 255.0
    dset.attrs = np.asarray(f['attrs'], 'float32')
    dset.attr_names = np.asarray(f['attr_names'])

    if size > 0:
        dset.images = dset.images[:size]
        dset.attrs = dset.attrs[:size]

    return dset
