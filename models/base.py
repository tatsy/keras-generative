import os
import sys
import numpy as np

from abc import ABCMeta, abstractmethod

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
            save_images(gan, samples, outfile)
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
