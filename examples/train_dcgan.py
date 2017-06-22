import os
import math
import argparse

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import keras
from keras.models import Model
from keras.datasets import mnist
from keras.layers import Input, Flatten, Dense, Reshape, Lambda
from keras.layers import BatchNormalization
from keras.layers import Activation, LeakyReLU
from keras.layers import Convolution2D, Deconvolution2D, UpSampling2D
from keras import backend as K

import h5py

from models import DCGAN

z_dims = 128

def load_celebA(filename):
    f = h5py.File(filename)
    return np.asarray(f['images'], np.float32)

def save_images(gen, samples, filename):
    imgs = gen.predict(samples)
    imgs = np.clip(imgs, 0.0, 1.0)
    if imgs.shape[3] == 1:
        imgs = np.squeeze(imgs, axis=(3,))

    fig = plt.figure(figsize=(8, 8))
    grid = gridspec.GridSpec(10, 10, wspace=0.1, hspace=0.1)
    for i in range(100):
        ax = plt.Subplot(fig, grid[i])
        if imgs.ndim == 4:
            ax.imshow(imgs[i, :, :, :], interpolation='none', vmin=0.0, vmax=1.0)
        else:
            ax.imshow(imgs[i, :, :], cmap='gray', interpolation='none', vmin=0.0, vmax=1.0)
        ax.axis('off')
        fig.add_subplot(ax)

    fig.savefig(filename, dpi=200)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Keras VAE')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=50)
    parser.add_argument('--output', default='output')

    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    gan = DCGAN(z_dims=z_dims)

    datasets = load_celebA('datasets/celebA.hdf5')
    datasets = datasets * 2.0 - 1.0

    # Training loop
    num_data = len(datasets)
    samples = np.random.normal(size=(100, z_dims)).astype(np.float32)
    for e in range(args.epoch):
        perm = np.arange(num_data, dtype=np.int32)
        np.random.shuffle(perm)
        for b in range(0, num_data, args.batchsize):
            batchsize = min(args.batchsize, num_data - b)
            indx = perm[b:b+batchsize]

            x_batch = datasets[indx, :, :, :]

            loss = gan.train_on_batch(x_batch)
            ratio = 100.0 * (b + batchsize) / num_data
            print(' epoch #%d :: %6.2f %% :: g_loss = %8.6f :: d_loss = %8.6f' % (e + 1, ratio, loss['g_loss'], loss['d_loss']),
                  end='\r', flush=True)

        # Show current generated images
        outfile = os.path.join(args.output, 'epoch_%04d.png' % (e + 1))
        save_images(gan, samples, outfile)
        print('')

if __name__ == '__main__':
    main()
