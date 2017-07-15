import os
import sys
import math
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib
matplotlib.use('Agg')

import keras
from keras.datasets import mnist

from models import CVAE, CVAEGAN, CALI, TripleGAN
from datasets import load_data, ConditionalDataset

models = {
    'cvae': CVAE,
    'cvaegan': CVAEGAN,
    'cali': CALI,
    'triple_gan': TripleGAN
}

def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Training GANs or VAEs')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=50)
    parser.add_argument('--datasize', type=int, default=-1)
    parser.add_argument('--output', default='output')
    parser.add_argument('--zdims', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--testmode', action='store_true')

    args = parser.parse_args()

    # Select GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Make output direcotiry if not exists
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # Load datasets
    if args.dataset == 'mnist':
        (images, attrs), _ = mnist.load_data()
        images = np.pad(images, ((0, 0), (2, 2), (2, 2)), 'constant', constant_values=0)
        images = (images[:, :, :, np.newaxis] / 255.0).astype('float32')
        attrs = keras.utils.to_categorical(attrs, 10).astype('float32')
        datasets = ConditionalDataset()
        datasets.images = images
        datasets.attrs = attrs
        datasets.attr_names = [str(i) for i in range(10)]
    else:
        datasets = load_data(args.dataset)

    # Construct model
    if args.model not in models:
        raise Exception('Unknown model:', args.model)

    model = models[args.model](
        input_shape=datasets.images.shape[1:],
        num_attrs=len(datasets.attr_names),
        z_dims=args.zdims,
        output=args.output
    )

    if args.resume is not None:
        model.load_model(args.resume)

    # Training loop
    datasets.images = datasets.images * 2.0 - 1.0
    samples = np.random.normal(size=(10, args.zdims)).astype(np.float32)
    model.main_loop(datasets, samples, datasets.attr_names,
        epochs=args.epoch,
        batchsize=args.batchsize,
        reporter=['loss', 'g_loss', 'd_loss', 'g_acc', 'd_acc', 'c_loss', 'ae_loss'])

if __name__ == '__main__':
    main()
