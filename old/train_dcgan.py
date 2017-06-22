import os
import sys
import math
import argparse

import numpy as np

import keras
from keras.models import Model
from keras.datasets import mnist
from keras.layers import Input, Flatten, Dense, Reshape, Lambda
from keras.layers import BatchNormalization
from keras.layers import Activation, LeakyReLU
from keras.layers import Convolution2D, Deconvolution2D, UpSampling2D
from keras import backend as K

root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root_folder)

from models import DCGAN
from basics import *

z_dims = 128

def main():
    parser = argparse.ArgumentParser(description='Keras VAE')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=50)
    parser.add_argument('--output', default='output')

    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    gan = DCGAN(z_dims=z_dims, name='dcgan', output=args.output)

    datasets = load_celebA('datasets/celebA.hdf5')
    datasets = datasets * 2.0 - 1.0

    # Training loop
    samples = np.random.normal(size=(100, z_dims)).astype(np.float32)
    gan.main_loop(datasets, samples,
        epochs=args.epoch,
        batchsize=args.batchsize,
        reporter=['g_loss', 'd_loss'])

if __name__ == '__main__':
    main()
