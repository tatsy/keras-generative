import os
import math
import argparse

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import keras
from keras.models import Model
from keras.datasets import mnist
from keras.layers import Input, Flatten, Dense, Reshape, Lambda, Concatenate
from keras.layers import BatchNormalization
from keras.layers import Activation, LeakyReLU
from keras.layers import Convolution2D, Deconvolution2D, UpSampling2D
from keras import backend as K

seed_dims = 50
num_digits = 10
image_shape = (28, 28, 1)

def normal(args):
    avg, logvar = args
    batch_size = K.shape(avg)[0]
    eps = K.random_normal(shape=(batch_size, seed_dims), mean=0.0, stddev=1.0)
    return avg + K.exp(logvar / 2.0) * eps

def Encoder():
    x_inputs = Input(shape=image_shape)
    y_inputs = Input(shape=(num_digits,))

    y = Reshape((1, 1, num_digits))(y_inputs)
    y = UpSampling2D(size=(image_shape[0], image_shape[1]))(y)

    x = Concatenate()([x_inputs, y])

    x = Convolution2D(filters=32, kernel_size=(5, 5))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Convolution2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Convolution2D(filters=64, kernel_size=(5, 5))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Convolution2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    avg = Dense(seed_dims)(x)
    log_var = Dense(seed_dims)(x)

    return Model([x_inputs, y_inputs], [avg, log_var], name='encoder')

def Generator():
    y_inputs = Input(shape=(num_digits,))
    z_inputs = Input(shape=(seed_dims,))

    x = Concatenate()([y_inputs, z_inputs])
    x = Dense(4 * 4 * 64)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Reshape((4, 4, 64))(x)

    x = Deconvolution2D(filters=64, kernel_size=(2, 2), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Deconvolution2D(filters=32, kernel_size=(5, 5))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Deconvolution2D(filters=32, kernel_size=(2, 2), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Deconvolution2D(filters=1, kernel_size=(5, 5))(x)
    x = Activation('sigmoid')(x)

    return Model([y_inputs, z_inputs], x, name='generator')

def variational_loss(avg, log_var):
    def lossfun(xy_true, x_pred):
        x_true = xy_true[0]
        size = K.shape(x_true)[1:]
        scale = K.cast(K.prod(size), 'float32')
        entropy = K.mean(keras.metrics.binary_crossentropy(x_true, x_pred[0])) * scale
        kl_loss = K.mean(-0.5 * K.sum(1.0 + log_var - K.square(avg) - K.exp(log_var), axis=-1))
        return entropy + kl_loss

    return lossfun

def main():
    parser = argparse.ArgumentParser(description='Keras VAE')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=50)

    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train[:, :, :, np.newaxis]
    x_test = x_test[:, :, :, np.newaxis]
    y_train = keras.utils.to_categorical(y_train, num_digits)
    y_test = keras.utils.to_categorical(y_test, num_digits)
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    enc = Encoder()
    enc.summary()
    gen = Generator()
    gen.summary()

    x_inputs = Input(shape=image_shape)
    y_inputs = Input(shape=(num_digits,))

    avg, log_var = enc([x_inputs, y_inputs])
    z = Lambda(normal, output_shape=(seed_dims,))([avg, log_var])
    x = gen([y_inputs, z])

    trainer = Model([x_inputs, y_inputs], x)
    trainer.compile(loss=variational_loss(avg, log_var),
                    optimizer='adadelta')
    trainer.summary()

    # Training loop
    num_data = len(y_train)
    z_samples = np.random.normal(size=(100, seed_dims)).astype(np.float32)
    y_samples = np.tile(np.eye(10, 10), [10, 1])

    for e in range(args.epoch):
        perm = np.arange(num_data, dtype=np.int32)
        np.random.shuffle(perm)
        for b in range(0, num_data, args.batchsize):
            batchsize = min(args.batchsize, num_data - b)
            indx = perm[b:b+batchsize]

            x_batch = x_train[indx, :, :, :]
            y_batch = y_train[indx]

            loss = trainer.train_on_batch([x_batch, y_batch], x_batch)
            ratio = 100.0 * (b + batchsize) / num_data
            print(' epoch #{:4d} | {:6.2f} % | {:8.6f}'.format(e + 1, ratio, loss), end='\r')

        print('')

        # Show current generated images
        imgs = gen.predict([y_samples, z_samples])
        fig, axs = plt.subplots(10, 10)
        for i in range(100):
            r = i // 10
            c = i % 10
            axs[r, c].imshow(imgs[i, :, :, 0], vmin=0.0, vmax=1.0, cmap='gray')
            axs[r, c].axis('off')

        fig.savefig('epoch_{:04d}.png'.format(e + 1))
        plt.close(fig)

if __name__ == '__main__':
    main()
