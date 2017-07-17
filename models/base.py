import os
import sys
import time
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras.models import load_model
from abc import ABCMeta, abstractmethod

from .utils import *

def time_format(t):
    m, s = divmod(t, 60)
    m = int(m)
    s = int(s)
    if m == 0:
        return '%d sec' % s
    else:
        return '%d min %d sec' % (m, s)

class BaseModel(metaclass=ABCMeta):
    '''
    Base class for non-conditional generative networks
    '''

    def __init__(self, **kwargs):
        '''
        Initialization
        '''
        if 'name' not in kwargs:
            raise Exception('Please specify model name!')

        self.name = kwargs['name']

        if 'input_shape' not in kwargs:
            raise Exception('Please specify input shape!')

        self.check_input_shape(kwargs['input_shape'])
        self.input_shape = kwargs['input_shape']

        if 'output' not in kwargs:
            self.output = 'output'
        else:
            self.output = kwargs['output']

        self.test_mode = False
        self.trainers = {}

    def check_input_shape(self, input_shape):
        # Check for CelebA
        if input_shape == (64, 64, 3):
            return

        # Check for MNIST (size modified)
        if input_shape == (32, 32, 1):
            return

        # Check for Cifar10, 100 etc
        if input_shape == (32, 32, 3):
            return

        errmsg = 'Input size should be 32 x 32 or 64 x 64!'
        raise Exception(errmsg)

    def main_loop(self, datasets, samples, epochs=100, batchsize=100, reporter=[]):
        '''
        Main learning loop
        '''
        # Create output directories if not exist
        out_dir = os.path.join(self.output, self.name)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        res_out_dir = os.path.join(out_dir, 'results')
        if not os.path.isdir(res_out_dir):
            os.mkdir(res_out_dir)

        wgt_out_dir = os.path.join(out_dir, 'weights')
        if not os.path.isdir(wgt_out_dir):
            os.mkdir(wgt_out_dir)

        # Start training
        print('\n\n--- START TRAINING ---\n')
        num_data = len(datasets)
        for e in range(epochs):
            perm = np.random.permutation(num_data)
            start_time = time.time()
            for b in range(0, num_data, batchsize):
                bsize = min(batchsize, num_data - b)
                indx = perm[b:b+bsize]

                # Get batch and train on it
                x_batch = self.make_batch(datasets, indx)
                losses = self.train_on_batch(x_batch)

                # Print current status
                ratio = 100.0 * (b + bsize) / num_data
                print(chr(27) + "[2K", end='')
                print('\rEpoch #%d | %d / %d (%6.2f %%) ' % \
                      (e + 1, b + bsize, num_data, ratio), end='')

                for k in reporter:
                    if k in losses:
                        print('| %s = %8.6f ' % (k, losses[k]), end='')

                # Compute ETA
                elapsed_time = time.time() - start_time
                eta = elapsed_time / (b + bsize) * (num_data - (b + bsize))
                print('| ETA: %s ' % time_format(eta), end='')

                sys.stdout.flush()

                # Save generated images
                if (b + bsize) % 10000 == 0 or (b+ bsize) == num_data:
                    outfile = os.path.join(res_out_dir, 'epoch_%04d_batch_%d.png' % (e + 1, b + bsize))
                    self.save_images(samples, outfile)

                if self.test_mode:
                    print('\nFinish testing: %s' % self.name)
                    return

            print('')

            # Save current weights
            self.save_model(wgt_out_dir, e + 1)

    def make_batch(self, datasets, indx):
        '''
        Get batch from datasets
        '''
        return datasets[indx]

    def save_images(self, samples, filename):
        '''
        Save images generated from random sample numbers
        '''
        imgs = self.predict(samples) * 0.5 + 0.5
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

    def save_model(self, out_dir, epoch):
        folder = os.path.join(out_dir, 'epoch_%05d' % epoch)
        if not os.path.isdir(folder):
            os.mkdir(folder)

        for k, v in self.trainers.items():
            filename = os.path.join(folder, '%s.hdf5' % (k))
            v.save_weights(filename)

    def store_to_save(self, name):
        self.trainers[name] = getattr(self, name)

    def load_model(self, folder):
        for k, v in self.trainers.items():
            filename = os.path.join(folder, '%s.hdf5' % (k))
            getattr(self, k).load_weights(filename)

    @abstractmethod
    def predict(self, z_sample):
        '''
        Plase override "predict" method in the derived model!
        '''
        pass


    @abstractmethod
    def train_on_batch(self, x_batch):
        '''
        Plase override "train_on_batch" method in the derived model!
        '''
        pass
