import h5py
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
