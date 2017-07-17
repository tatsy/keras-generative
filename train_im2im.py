import os
import sys
import math
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
matplotlib.use('Agg')
import numpy as np

from models import *
from datasets import *

svhn.load_data()
sys.exit(0)

models = {
    'cycle_gan': CycleGAN,
    'unit': UNIT
}

class PairwiseDataset(object):
    def __init__(self):
        self.x_datasets = None
        self.y_datasets = None

    def __len__(self):
        return len(self.x_datasets)

def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Training GANs or VAEs')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--datasets', type=str, required=True)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--batchsize', type=int, default=50)
    parser.add_argument('--output', default='output')
    parser.add_argument('--zdims', type=int, default=128)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    # select gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Make output direcotiry if not exists
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # Construct model
    if args.model not in models:
        raise Exception('Unknown model:', args.model)

    model = models[args.model](z_dims=args.zdims, output=args.output)

    if args.resume is not None:
        model.load_model(args.resume)

    x_datafile = os.path.join(args.datasets, 'celebA.hdf5')
    x_datasets = load_data(x_datafile).images
    x_datasets = x_datasets * 2.0 - 1.0
    x_num_data = len(x_datasets)

    y_datafile = os.path.join(args.datasets, 'animeface.hdf5')
    y_datasets = load_data(y_datafile).images
    y_datasets = y_datasets * 2.0 - 1.0
    y_num_data = len(y_datasets)

    datasets = PairwiseDataset()
    num_data = min(x_num_data, y_num_data)
    datasets.x_datasets = x_datasets[:num_data]
    datasets.y_datasets = y_datasets[:num_data]

    samples = x_datasets[num_data:num_data+50]

    # Training loop
    model.main_loop(datasets, samples,
        epochs=args.epoch,
        batchsize=args.batchsize,
        reporter=['loss', 'g_loss', 'd_loss', 'g_acc', 'd_acc'])

if __name__ == '__main__':
    main()
