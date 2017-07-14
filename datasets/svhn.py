import os
import requests

import numpy as np
import scipy as sp
import scipy.io

import keras

from .datasets import ConditionalDataset

url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
curdir = os.path.abspath(os.path.dirname(__file__))
outdir = os.path.join(curdir, 'files')
outfile = os.path.join(outdir, 'svhn.mat')

CHUNK_SIZE = 32768

def download_svhn():
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    session = requests.Session()
    response = session.get(url, stream=True)
    with open(outfile, 'wb') as fp:
        dl = 0
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                dl += len(chunk)
                fp.write(chunks)

                mb = dl / 1.0e6
                sys.stdout.write('\r%.2f MB downloaded...' % (mb))
                sys.stdout.flush()

        sys.stdout.write('\nFinish!\n')
        sys.stdout.flush()

def load_data():
    if not os.path.exists(outfile):
        download_svhn()

    mat = sp.io.loadmat(outfile)
    x_train = mat['X']
    y_train = mat['y']

    x_train = np.transpose(x_train, axes=[3, 0, 1, 2])
    x_train = (x_train / 255.0).astype('float32')
    y_train[y_train == 10] = 0
    y_train = keras.utils.to_categorical(y_train)
    y_train = y_train.astype('float32')

    datasets = ConditionalDataset()
    datasets.images = x_train
    datasets.attrs = y_train
    datasets.attr_names = [str(i) for i in range(10)]

    return datasets
