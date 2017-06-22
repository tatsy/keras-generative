import os
import sys
import math
import argparse

import numpy as np

from models import VAE, DCGAN, EBGAN
from basics import *

z_dims = 128

models = {
    'vae': VAE,
    'dcgan': DCGAN,
    'ebgan': EBGAN
}

def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Training GANs or VAEs')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=50)
    parser.add_argument('--output', default='output')

    args = parser.parse_args()

    # Make output direcotiry if not exists
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # Construct model
    if args.model not in models:
        raise Exception('Unknown model:', args.model)

    model = models[args.model](z_dims=z_dims, output=args.output)

    datasets = load_celebA('datasets/celebA.hdf5')
    datasets = datasets * 2.0 - 1.0

    # Training loop
    samples = np.random.normal(size=(100, z_dims)).astype(np.float32)
    model.main_loop(datasets, samples,
        epochs=args.epoch,
        batchsize=args.batchsize,
        reporter=['loss', 'g_loss', 'd_loss'])

if __name__ == '__main__':
    main()
