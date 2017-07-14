import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from abc import ABCMeta, abstractmethod

class Im2imBaseModel(BaseModel):
    def __init__(self, **kwargs):
        super(Im2imBaseModel, self).__init__(**kwargs)
        self.attr_names = None

    def save_images(self, gen, samples, filename):
        '''
        Save images generated from random sample numbers
        '''
        imgs = gen.predict(samples) * 0.5 + 0.5
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
