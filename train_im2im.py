import os
import sys
import math
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
matplotlib.use('Agg')
import numpy as np

from models import *
import datasets as dsets
from datasets import PairwiseDataset

models = {
    'cyclegan': CycleGAN,
    'unit': UNIT
}

def load_data(data_type):
    if data_type == 'mnist':
        return dsets.mnist.load_data()
    elif data_type == 'svhn':
        return dsets.svhn.load_data()
    else:
        return dsets.load_data(data_type)

def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Training GANs or VAEs')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--first-data', type=str, required=True)
    parser.add_argument('--second-data', type=str, required=True)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--batchsize', type=int, default=50)
    parser.add_argument('--output', default='output')
    parser.add_argument('--zdims', type=int, default=128)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--testmode', action='store_true')

    args = parser.parse_args()

    # select gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Make output direcotiry if not exists
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # Load datasets
    x_data = load_data(args.first_data)
    x_data = x_data.images * 2.0 - 1.0
    y_data = load_data(args.second_data)
    y_data = y_data.images * 2.0 - 1.0

    datasets = PairwiseDataset(x_data, y_data)
    num_data = len(datasets)

    # Construct model
    if args.model not in models:
        raise Exception('Unknown model:', args.model)

    model = models[args.model](
        input_shape=datasets.shape[1:],
        z_dims=args.zdims,
        output=args.output
    )

    if args.resume is not None:
        model.load_model(args.resume)

    # Make samples
    x_samples = datasets.x_data[num_data:num_data+25]
    y_samples = datasets.y_data[num_data:num_data+25]
    samples = (x_samples, y_samples)

    # Training loop
    model.main_loop(datasets, samples,
        epochs=args.epoch,
        batchsize=args.batchsize,
        reporter=['loss', 'g_loss', 'd_loss', 'g_acc', 'd_acc'])

if __name__ == '__main__':
    main()
