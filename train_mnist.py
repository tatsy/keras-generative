import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import keras
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.layers import Input, Flatten, Dense, Activation, Reshape
from keras.layers import Convolution2D, Deconvolution2D, MaxPooling2D, GlobalAveragePooling2D, LeakyReLU, BatchNormalization
from keras import backend as K

seed_dims = 100

def Generator():
    inputs = Input(shape=(seed_dims,))
    x = Dense(4 * 4 * 64)(inputs)
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
    x = Activation('tanh')(x)

    model = Model(inputs, x, name='generator')

    return model

def Discriminator():
    inputs = Input(shape=(28, 28, 1))
    x = Convolution2D(filters=32, kernel_size=(5, 5))(inputs)
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

    x = Convolution2D(filters=2, kernel_size=(1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)

    model = Model(inputs, x, name='discriminator')
    return model

def set_trainable(model, train):
    model.trainable = train
    for l in model.layers:
        l.trainable = train

def main():
    parser = argparse.ArgumentParser(description='Keras DCGAN')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=50)
    parser.add_argument('--output', default='output')
    parser.add_argument('--result', default='result')

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    if not os.path.exists(args.result):
        os.mkdir(args.result)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 127.5 - 1.0
    x_test = x_test / 127.5 - 1.0
    x_train = x_train[:, :, :, np.newaxis]
    x_test = x_test[:, :, :, np.newaxis]
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    gen = Generator()
    gen.summary()
    dis = Discriminator()
    dis.summary()

    # gen_optim = keras.optimizers.Adam(lr=2e-4, beta_1=0.5)
    # dis_optim = keras.optimizers.Adam(lr=1e-5, beta_1=0.1)
    gen_optim = keras.optimizers.Adadelta()
    dis_optim = keras.optimizers.Adadelta()

    # Trainer for generator
    set_trainable(gen, True)
    set_trainable(dis, False)
    gen_input = Input(shape=(seed_dims,))
    gen_x = gen(gen_input)
    gen_x = dis(gen_x)
    gen_trainer = Model(gen_input, gen_x)
    gen_trainer.compile(loss=keras.losses.binary_crossentropy,
                        optimizer=gen_optim,
                        metrics=['accuracy'])

    # Trainer for discriminator
    set_trainable(gen, False)
    set_trainable(dis, True)
    dis_input = Input(shape=(seed_dims,))
    dis_x = gen(dis_input)
    dis_x = dis(dis_x)

    data_input = Input(shape=(28, 28, 1))
    data_x = dis(data_input)
    dis_trainer = Model(inputs=[dis_input, data_input], outputs=[dis_x, data_x])
    dis_trainer.compile(loss=keras.losses.binary_crossentropy,
                        optimizer=dis_optim,
                        metrics=['accuracy'])

    # Training loop
    print(' {:10s} | {:8s} | {:8s} | {:8s} | {:8s} | {:8s} | {:8s} | {:8s}'.format(
        'epoch', 'done', 'gen loss', 'gen acc', 'dis loss', 'dis acc', 'loss', 'acc'))

    num_data = len(y_train)
    samples = np.random.uniform(-1.0, 1.0, size=(100, seed_dims))
    for e in range(args.epoch):
        perm = np.arange(num_data, dtype=np.int32)
        np.random.shuffle(perm)
        for b in range(0, num_data, args.batchsize):
            batchsize = min(args.batchsize, num_data - b)
            indx = perm[b:b+batchsize]

            y_pos = np.ones(batchsize, np.int32)
            y_pos = keras.utils.to_categorical(y_pos, 2)
            y_neg = np.zeros(batchsize, np.int32)
            y_neg = keras.utils.to_categorical(y_neg, 2)

            x_batch = x_train[indx, :, :, :]
            rnd_batch = np.random.uniform(-1.0, 1.0, size=(batchsize, seed_dims)).astype(np.float32)

            gen_loss, gen_acc = gen_trainer.train_on_batch(rnd_batch, y_pos)
            _, dis_loss, loss, dis_acc, acc = dis_trainer.train_on_batch([rnd_batch, x_batch], [y_neg, y_pos])

            ratio = min(100.0, 100.0 * (b + batchsize + 1) / num_data)
            print(' epoch #{:d} | {:6.2f} % | {:8.6f} | {:8.6f} | {:8.6f} | {:8.6f} | {:8.6f} | {:8.6f}'.format(
                e + 1, ratio, gen_loss, gen_acc, dis_loss, dis_acc, loss, acc), end='\r')

        print('')

        # Save model
        if (e + 1) % 10 == 0:
            dis.save_weights(os.path.join(args.result, 'weights_discriminator_epoch_{:04d}.hdf5'.format(e + 1)))
            gen.save_weights(os.path.join(args.result, 'weights_generator_epoch_{:04d}.hdf5'.format(e + 1)))

        # Show current generated images
        imgs = gen.predict(samples) * 0.5 + 0.5
        fig, axs = plt.subplots(10, 10)
        for i in range(100):
            r = i // 10
            c = i % 10
            axs[r, c].imshow(imgs[i, :, :, 0], vmin=0.0, vmax=1.0, cmap='gray')
            axs[r, c].axis('off')

        fig.savefig(os.path.join(args.output, 'epoch_{:04d}.png'.format(e + 1)))
        plt.close(fig)

if __name__ == '__main__':
    main()
