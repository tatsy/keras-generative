import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from abc import ABCMeta, abstractmethod

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

class BaseModel(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        if 'name' not in kwargs:
            raise Exception('Please specify model name!')

        self.name = kwargs['name']

        if 'output' not in kwargs:
            self.output = 'output'
        else:
            self.output = kwargs['output']

    def main_loop(self, datasets, samples, epochs=100, batchsize=100, reporter=[]):
        print('\n\n--- START TRAINING ---\n')
        num_data = len(datasets)
        for e in range(epochs):
            perm = np.arange(num_data, dtype=np.int32)
            np.random.shuffle(perm)
            for b in range(0, num_data, batchsize):
                bsize = min(batchsize, num_data - b)
                indx = perm[b:b+bsize]

                x_batch = datasets[indx, :, :, :]

                loss = self.train_on_batch(x_batch)
                ratio = 100.0 * (b + bsize) / num_data
                print('\rEpoch #%d | %d / %d (%6.2f %%) ' % (e + 1, b + bsize, num_data, ratio), end='')
                for k in reporter:
                    if k in loss:
                        print('| %s = %8.6f ' % (k, loss[k]), end='')

                sys.stdout.flush()

            # Save generated images
            out_dir = os.path.join(self.output, self.name)
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)

            res_out_dir = os.path.join(out_dir, 'results')
            if not os.path.isdir(res_out_dir):
                os.mkdir(res_out_dir)

            outfile = os.path.join(res_out_dir, 'epoch_%04d.png' % (e + 1))
            save_images(self, samples, outfile)
            print('')

            # Save current weights
            wgt_out_dir = os.path.join(out_dir, 'weights')
            if not os.path.isdir(wgt_out_dir):
                os.mkdir(wgt_out_dir)

            self.save_weights(wgt_out_dir, e + 1, b + bsize)

    @abstractmethod
    def train_on_batch(self, x_batch):
        print('No training process is defined! Plase override "train_on_batch" method in the derived model!')

    @abstractmethod
    def save_weights(self, out_dir, epoch, batch):
        print('Model weights are not saved! To save them, override the "save_weights" method in the derived model.')
