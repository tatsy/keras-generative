import os
import sys
import math
import struct
import argparse

import numpy as np
import scipy.misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Activation, Reshape, Dropout, Concatenate, Lambda, GaussianNoise
from keras.layers import Convolution2D, Deconvolution2D, LeakyReLU, ELU, BatchNormalization
from keras.layers import UpSampling2D, AveragePooling2D, GlobalAveragePooling2D
from keras import backend as K

z_dims = 50
h_dims = 512
n_filters = 32
image_shape = (28, 28, 1)

def Decoder(input_dims):
    inputs = Input(shape=(input_dims,))

    # Fully connected layer
    x = Dense(4 * 4 * n_filters)(inputs)
    x = Activation('relu')(x)
    x = Reshape((4, 4, n_filters))(x)

    # Layer 1
    x = Convolution2D(filters=n_filters, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(filters=n_filters, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D(size=(2, 2))(x)

    # Layer 2
    x = Convolution2D(filters=n_filters, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(filters=n_filters, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D(size=(2, 2))(x)

    # Layer 3
    x = Convolution2D(filters=n_filters, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(filters=n_filters, kernel_size=(3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D(size=(2, 2))(x)

    # Layer 4
    x = Convolution2D(filters=n_filters, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Convolution2D(filters=image_shape[2], kernel_size=(3, 3), padding='same')(x)
    x = Activation('tanh')(x)

    model = Model(inputs, x)
    return model

def Encoder():
    inputs = Input(shape=image_shape)

    # Add noise
    x = GaussianNoise(stddev=0.3)(inputs)

    # Layer 1
    x = Convolution2D(filters=n_filters, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = Convolution2D(filters=n_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    # Layer 2
    x = Convolution2D(filters=n_filters * 2, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = Convolution2D(filters=n_filters * 2, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    # Layer 3
    x = Convolution2D(filters=n_filters * 3, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = Convolution2D(filters=n_filters * 3, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    # Fully connected layer
    x = Flatten()(x)
    x = Dense(h_dims)(x)
    x = Activation('linear')(x)

    model = Model(inputs, x)
    return model

"""
seed_dims = 100
n_filters = 128
image_shape = (64, 64, 3)

def Decoder():
    inputs = Input(shape=(seed_dims,))

    # Fully connected layer
    x = Dense(8 * 8 * n_filters)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Reshape((8, 8, n_filters))(x)

    # Layer 1
    x = Deconvolution2D(filters=n_filters, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Deconvolution2D(filters=n_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Layer 2
    x = Deconvolution2D(filters=n_filters, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Deconvolution2D(filters=n_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Layer 3
    x = Deconvolution2D(filters=n_filters, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Deconvolution2D(filters=n_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Layer 4
    x = Deconvolution2D(filters=n_filters, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Deconvolution2D(filters=3, kernel_size=(3, 3), padding='same')(x)
    x = Activation('tanh')(x)

    model = Model(inputs, x)
    return model

def Encoder():
    inputs = Input(shape=image_shape)

    # Layer 1
    x = Convolution2D(filters=n_filters, kernel_size=(3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = Convolution2D(filters=n_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    # Layer 2
    x = Convolution2D(filters=n_filters * 2, kernel_size=(3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = Convolution2D(filters=n_filters * 2, kernel_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    # Layer 3
    x = Convolution2D(filters=n_filters * 3, kernel_size=(3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = Convolution2D(filters=n_filters * 3, kernel_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    # Layer 4
    x = Convolution2D(filters=n_filters * 4, kernel_size=(3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = Convolution2D(filters=n_filters * 4, kernel_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    # Fully connected layer
    x = Flatten()(x)
    x = Dense(seed_dims)(x)
    x = Activation('tanh')(x)

    model = Model(inputs, x)
    return model
"""

class GeneratorLoss(object):
    __name__ = 'generator_loss'

    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        x_random, y_random = y_pred[:, :, :, 0:image_shape[2]], y_pred[:, :, :, image_shape[2]:image_shape[2]*2]
        gen_loss = K.mean(K.abs(x_random - y_random), axis=[1, 2, 3])
        return gen_loss

class DiscriminatorLoss(object):
    __name__ = 'discriminator_loss'

    def __init__(self, gamma, lambda_k):
        self.lambda_k = lambda_k
        self.gamma = gamma
        self.k_t = K.variable(0.0, dtype=K.floatx())
        self.updates = []

    def __call__(self, y_true, y_pred):
        x_random, x_data = y_true[:, :, :, 0:image_shape[2]], y_true[:, :, :, image_shape[2]:image_shape[2]*2]
        y_random, y_data = y_pred[:, :, :, 0:image_shape[2]], y_pred[:, :, :, image_shape[2]:image_shape[2]*2]

        gen_loss = K.mean(K.abs(x_random - y_random), axis=[1, 2, 3])
        dis_loss = K.mean(K.abs(x_data - y_data), axis=[1, 2, 3])
        loss = dis_loss - self.k_t * gen_loss

        mean_gen_loss = K.mean(gen_loss)
        mean_dis_loss = K.mean(dis_loss)
        # self.gamma = mean_gen_loss / (mean_dis_loss + 1.0e-12)
        # self.gamma = K.clip(self.gamma, 0.0, 1.0)
        new_k_t = self.k_t + self.lambda_k * (self.gamma * mean_dis_loss - mean_gen_loss)
        new_k_t = K.clip(new_k_t, 0.0, 1.0)

        self.updates.append(K.update(self.k_t, new_k_t))

        return loss

class DiscriminatorModel(Model):
    @property
    def updates(self):
        updates = super(DiscriminatorModel, self).updates
        if hasattr(self, 'loss_functions'):
            for loss_func in self.loss_functions:
                if hasattr(loss_func, 'updates'):
                    updates += loss_func.updates

        return updates

def save_images(gen, samples, output, epoch, batch=-1):
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

    outfile = os.path.join(output, 'epoch_{:04d}.png'.format(epoch + 1))
    if batch >= 0:
        outfile = os.path.join(output, 'epoch_{:04d}-{:d}.png'.format(epoch + 1, batch))

    fig.savefig(outfile, dpi=200)
    plt.close(fig)

def set_trainable(model, train):
    model.trainable = train
    for l in model.layers:
        l.trainable = train

def progress_bar(x, maxval, width=40):
    tick = int(x / maxval * width)
    tick = min(tick, width)

    if tick == width:
        return '=' * tick

    return '=' * tick + '>' + ' ' * (width - tick - 1)

def load_data(folder, num_images=60000):
    (x_train, y_train), _ = mnist.load_data()
    return x_train

    files = [f for f in os.listdir(folder) if not f.startswith('.')]
    files = [os.path.join(folder, f) for f in files if f.endswith('.jpg')]

    if len(files) >= num_images:
        files = files[:num_images]

    print('Loading images...')
    n_images = len(files)
    data = [None] * n_images
    for i, f in enumerate(files):
        data[i] = scipy.misc.imread(f, mode='RGB')
        if data[i].ndim == 2:
            data[i] = np.tile(data[i][:, :, np.newaxis], [1, 3])
        ratio = 100.0 * (i + 1) / n_images
        print('[ {:6.2f} % ] [ {} ]'.format(ratio, progress_bar(i + 1, n_images)), end='\r', flush=True)

    print('')

    print(n_images, 'images loaded!')
    return np.stack(data, axis=0)

def build_generator(gen, enc, dec, optim):
    set_trainable(gen, True)
    set_trainable(enc, False)
    set_trainable(dec, False)

    h_random_input = Input(shape=(z_dims,))
    x_random = gen(h_random_input)

    h_random = enc(x_random)
    y_random = dec(h_random)

    all_output = Concatenate(axis=-1)([x_random, y_random])

    gen_trainer = Model(h_random_input, all_output)
    gen_trainer.summary()

    gen_trainer.compile(loss=GeneratorLoss(),
                        optimizer=optim)

    return gen_trainer

def build_discriminator(gen, enc, dec, optim):
    set_trainable(gen, False)
    set_trainable(enc, True)
    set_trainable(dec, True)

    all_input = Input(shape=(image_shape[0], image_shape[1], image_shape[2] * 2))

    x_random = Lambda(lambda x: x[:, :, :, 0:image_shape[2]], output_shape=image_shape)(all_input)
    x_data = Lambda(lambda x: x[:, :, :, image_shape[2]:image_shape[2]*2], output_shape=image_shape)(all_input)

    h_data = enc(x_data)
    y_data = dec(h_data)

    h_random = enc(x_random)
    y_random = dec(h_random)

    all_output = Concatenate(axis=-1)([y_random, y_data])

    dis_trainer = DiscriminatorModel(inputs=all_input, outputs=all_output)
    dis_trainer.summary()

    lambda_k = np.float(0.001)
    gamma = np.float(0.5)
    dis_trainer.compile(loss=DiscriminatorLoss(gamma, lambda_k),
                        optimizer=optim)

    return dis_trainer

def main():
    parser = argparse.ArgumentParser(description='Keras DCGAN')
    parser.add_argument('--data', required=True)
    parser.add_argument('--numtrain', type=int, default=60000)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=50)
    parser.add_argument('--output', default='output')
    parser.add_argument('--result', default='result')

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    if not os.path.exists(args.result):
        os.mkdir(args.result)

    x_train = load_data(args.data, args.numtrain)
    x_train = x_train.astype('float32') / 255.0
    x_train = x_train * 2.0 - 1.0
    x_train = x_train[:, :, :, np.newaxis]

    gen = Decoder(input_dims=z_dims)
    enc = Encoder()
    dec = Decoder(input_dims=h_dims)

    gen_optim = keras.optimizers.Adam(lr=2.0e-4, beta_1=0.5)
    dis_optim = keras.optimizers.Adam(lr=2.0e-4, beta_1=0.5)
    # gen_optim = keras.optimizers.Adadelta()
    # dis_optim = keras.optimizers.Adadelta()

    gen_trainer = build_generator(gen, enc, dec, gen_optim)
    dis_trainer = build_discriminator(gen, enc, dec, dis_optim)

    # Training loop
    num_data = len(x_train)
    samples = np.random.uniform(-1.0, 1.0, size=(100, z_dims)).astype(np.float32)
    for e in range(args.epoch):
        perm = np.random.permutation(num_data)
        gen_loss_sum = np.float32(0.0)
        dis_loss_sum = np.float32(0.0)

        num_batches = (num_data + args.batchsize - 1) // args.batchsize
        for b in range(num_batches):
            batch_start = b * args.batchsize
            batch_end = min(batch_start + args.batchsize, num_data)
            batch_size = batch_end - batch_start
            indx = perm[batch_start:batch_end]

            h_rand_batch = np.random.uniform(-1.0, 1.0, size=(batch_size, z_dims)).astype(np.float32)
            x_rand_batch = gen.predict_on_batch(h_rand_batch)
            x_batch = x_train[indx, :, :, :]
            x_concat_batch = np.concatenate([x_rand_batch, x_batch], axis=-1)

            dis_loss = dis_trainer.train_on_batch(x_concat_batch, x_concat_batch)

            dummy_output = np.zeros(x_batch.shape)
            gen_loss = gen_trainer.train_on_batch(h_rand_batch, dummy_output)

            # Show info
            gen_loss_sum += gen_loss
            dis_loss_sum += dis_loss
            if e == 0 and b == 0:
                print(' {:10s} | {:8s} | {:12s} | {:12s} '.format(
                    'epoch', 'done', 'gen loss', 'dis loss'))

            ratio = min(100.0, 100.0 * batch_end / num_data)
            print(' epoch #{:3s} | {:6.2f} % | {:12.6f} | {:12.6f} '.format(
                '%d' % (e + 1), ratio,
                gen_loss_sum / (b + 1), dis_loss_sum / (b + 1)), end='\r')

            if (b + 1) % 200 == 0:
                save_images(gen, samples, args.output, e, batch_end)

        print('')

        # Save model
        if (e + 1) % 10 == 0:
            gen.save_weights(os.path.join(args.result, 'weights_generator_epoch_{:04d}.hdf5'.format(e + 1)))
            enc.save_weights(os.path.join(args.result, 'weights_encoder_epoch_{:04d}.hdf5'.format(e + 1)))
            dec.save_weights(os.path.join(args.result, 'weights_decoder_epoch_{0:04d}.hdf5'.format(e + 1)))
            print('Current weights are saved in "{}"'.format(args.result))

        # Show current generated images
        save_images(gen, samples, args.output, e)

if __name__ == '__main__':
    main()
