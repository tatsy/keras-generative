import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from abc import ABCMeta, abstractmethod

from .base import BaseModel

class Im2imBaseModel(BaseModel):
    def __init__(self, **kwargs):
        super(Im2imBaseModel, self).__init__(**kwargs)

    def make_batch(self, datasets, indx):
        x = datasets.x_data[indx]
        y = datasets.y_data[indx]
        return (x, y)

    @abstractmethod
    def predict_x2y(self, x_sample):
        '''
        Plase override "predict_x2y" method in the derived model!
        '''
        pass

    @abstractmethod
    def predict_y2x(self, y_sample):
        '''
        Plase override "predict_y2x" method in the derived model!
        '''
        pass

    def save_images(self, samples, filename):
        x_samples, y_samples = samples

        x_img = self.predict_y2x(y_samples) * 0.5 + 0.5
        x_img = np.clip(x_img, 0.0, 1.0)
        y_img = self.predict_2xy(x_samples) * 0.5 + 0.5
        y_img = np.clip(y_img, 0.0, 1.0)

        images = np.concatenate(zip(x_samples, y_img), zip(y_samples, x_img))
        assert len(images) == 100

        fig = plt.figure(figsize=(8, 8))
        grid = gridspec.GridSpec(10, 10, wspace=0.1, hspace=0.1)
        for i in range(100):
            ax = plt.Subplot(fig, grid[i])
            if images.ndim == 4:
                ax.imshow(imgages[i, :, :, :], interpolation='none', vmin=0.0, vmax=1.0)
            else:
                ax.imshow(images[i, :, :], cmap='gray', interpolation='none', vmin=0.0, vmax=1.0)
            ax.axis('off')
            fig.add_subplot(ax)

        fig.savefig(filename, dpi=200)
        plt.close(fig)
