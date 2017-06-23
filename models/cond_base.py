import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .base import BaseModel

class CondBaseModel(BaseModel):
    def __init__(self, **kwargs):
        super(CondBaseModel, self).__init__(**kwargs)

        self.attr_names = None

    def main_loop(self, datasets, samples, attr_names, epochs=100, batchsize=100, reporter=[]):
        self.attr_names = attr_names
        super(CondBaseModel, self).main_loop(datasets, samples, epochs, batchsize, reporter)

    def make_batch(self, datasets, indx):
        images = datasets.images[indx]
        attrs = datasets.attribs[indx]
        return images, attrs

    def save_images(self, gen, samples, filename):
        assert self.attr_names is not None

        num_samples = len(samples)
        attrs = np.identity(self.num_attrs)
        attrs = np.tile(attrs, (num_samples, 1))

        samples = np.tile(samples, (1, self.num_attrs))
        samples = samples.reshape((num_samples * self.num_attrs, -1))

        imgs = gen.predict([samples, attrs])
        imgs = np.clip(imgs, 0.0, 1.0)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))

        fig = plt.figure(figsize=(32, 8))
        grid = gridspec.GridSpec(num_samples, self.num_attrs, wspace=0.1, hspace=0.1)
        for i in range(num_samples * self.num_attrs):
            ax = plt.Subplot(fig, grid[i])
            if imgs.ndim == 4:
                ax.imshow(imgs[i, :, :, :], interpolation='none', vmin=0.0, vmax=1.0)
            else:
                ax.imshow(imgs[i, :, :], cmap='gray', interpolation='none', vmin=0.0, vmax=1.0)
            ax.axis('off')
            fig.add_subplot(ax)

        fig.savefig(filename, dpi=200)
        plt.close(fig)
