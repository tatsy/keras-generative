import os
import sys
import math
import struct
import argparse

import numpy as np
import scipy.misc
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Activation, Reshape, Dropout
from keras.layers import Convolution2D, Deconvolution2D, LeakyReLU, BatchNormalization
from keras.layers import UpSampling2D, AveragePooling2D, GlobalAveragePooling2D
from keras import backend as K

seed_dims = 100

def Generator():
    inputs = Input(shape=(seed_dims,))
    x = Dense(4 * 4 * 1024, kernel_initializer='glorot_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Reshape((4, 4, 1024))(x)

    x = Deconvolution2D(filters=512, kernel_size=(5, 5),
                        kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    x = Activation('relu')(x)

    x = Deconvolution2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding='same',
                        kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    x = Activation('relu')(x)

    x = Deconvolution2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same',
                        kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    x = Activation('relu')(x)

    x = Deconvolution2D(filters=3, kernel_size=(5, 5), strides=(2, 2), padding='same',
                        kernel_initializer='glorot_normal')(x)
    x = Activation('tanh')(x)

    model = Model(inputs, x, name='generator')

    return model

def Discriminator():
    inputs = Input(shape=(64, 64, 3))
    x = Convolution2D(filters=128, kernel_size=(5, 5), padding='same',
                      kernel_initializer='glorot_normal')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = Convolution2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding='same',
                      kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = Convolution2D(filters=512, kernel_size=(5, 5), strides=(2, 2), padding='same',
                      kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = Convolution2D(filters=1024, kernel_size=(5, 5), strides=(2, 2), padding='same',
                      kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = Convolution2D(filters=2, kernel_size=(5, 5), strides=(2, 2), padding='same',
                      kernel_initializer='glorot_normal')(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)

    model = Model(inputs, x, name='discriminator')
    return model

def set_trainable(model, train):
    model.trainable = train
    for l in model.layers:
        l.trainable = train

def load_ppm(f):
    fp = open(f, 'rb')

    tag = fp.readline().decode('ascii').strip()
    h = int(fp.readline().decode('ascii'))
    w = int(fp.readline().decode('ascii'))
    maxval = int(fp.readline().decode('ascii'))

    data = struct.unpack('B' * h * w * 3, fp.read(h * w * 3))
    fp.close()

    return np.asarray(data, dtype=np.uint8).reshape((h, w, 3))

def progress_bar(x, maxval, width=40):
    tick = int(x / maxval * width)
    tick = min(tick, width)

    if tick == width:
        return '=' * tick

    return '=' * tick + '>' + ' ' * (width - tick - 1)

def load_data(folder, num_images=60000):
    files = [f for f in os.listdir(folder) if not f.startswith('.')]
    files = [os.path.join(folder, f) for f in files if f.endswith('.jpg')]

    if len(files) >= num_images:
        files = files[:num_images]

    print('Loading images...')
    n_images = len(files)
    data = [None] * n_images
    for i, f in enumerate(files):
        data[i] = scipy.misc.imread(f, mode='RGB')
        ratio = 100.0 * (i + 1) / n_images
        print('[ {:6.2f} % ] [ {} ]'.format(ratio, progress_bar(i + 1, n_images)), end='\r', flush=True)

    print('')

    print(n_images, 'images loaded!')
    return np.stack(data, axis=0)

def save_images(gen, samples, output, epoch, batch=-1):
    imgs = gen.predict(samples) * 0.5 + 0.5
    imgs = np.clip(imgs, 0.0, 1.0)

    fig = plt.figure(figsize=(8, 8))
    grid = gridspec.GridSpec(10, 10, wspace=0.1, hspace=0.1)
    for i in range(100):
        ax = plt.Subplot(fig, grid[i])
        ax.imshow(imgs[i, :, :, :], interpolation='none', vmin=0.0, vmax=1.0)
        ax.axis('off')
        fig.add_subplot(ax)

    outfile = os.path.join(output, 'epoch_{:04d}.png'.format(epoch + 1))
    if batch >= 0:
        outfile = os.path.join(output, 'epoch_{:04d}-{:d}.png'.format(epoch + 1, batch))

    fig.savefig(outfile, dpi=200)
    plt.close(fig)

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

    gen = Generator()
    gen.summary()
    dis = Discriminator()
    dis.summary()

    gen_optim = keras.optimizers.Adam(lr=1.0e-4, beta_1=0.5)
    dis_optim = keras.optimizers.Adam(lr=1.0e-4, beta_1=0.5)
    # gen_optim = keras.optimizers.Adadelta()
    # dis_optim = keras.optimizers.Adadelta()

    # Trainer for discriminator
    dis_input = Input(shape=(64, 64, 3))
    dis_x = dis(dis_input)
    dis_trainer = Model(inputs=dis_input, outputs=dis_x)
    dis_trainer.compile(loss=keras.losses.binary_crossentropy,
                        optimizer=dis_optim,
                        metrics=['accuracy'])

    # Trainer for generator
    set_trainable(dis, False)
    gen_input = Input(shape=(seed_dims,))
    gen_x = gen(gen_input)
    gen_x = dis(gen_x)
    gen_trainer = Model(gen_input, gen_x)
    gen_trainer.compile(loss=keras.losses.binary_crossentropy,
                        optimizer=gen_optim,
                        metrics=['accuracy'])

    # Training loop
    num_data = len(x_train)
    samples = np.random.uniform(-1.0, 1.0, size=(100, seed_dims)).astype(np.float32)
    for e in range(args.epoch):
        perm = np.random.permutation(num_data)
        gen_loss_sum = np.float32(0.0)
        gen_acc_sum = np.float32(0.0)
        dis_loss_sum = np.float32(0.0)
        dis_acc_sum = np.float32(0.0)

        num_batches = (num_data + args.batchsize - 1) // args.batchsize
        for b in range(num_batches):
            batch_start = b * args.batchsize
            batch_end = min(batch_start + args.batchsize, num_data)
            batch_size = batch_end - batch_start
            indx = perm[batch_start:batch_end]

            # Positive/negative labels
            # !! practically reversed labeling works well !!
            y_pos = np.zeros(batch_size, dtype=np.int32)
            y_pos = keras.utils.to_categorical(y_pos, 2)
            y_neg = np.ones(batch_size, dtype=np.int32)
            y_neg = keras.utils.to_categorical(y_neg, 2)

            x_batch = x_train[indx, :, :, :]
            rnd_batch = np.random.uniform(-1.0, 1.0, size=(batch_size, seed_dims)).astype(np.float32)

            gen_loss, gen_acc = gen_trainer.train_on_batch(rnd_batch, y_pos)

            rnd_pred = gen.predict_on_batch(rnd_batch)
            dis_loss, dis_acc = dis_trainer.train_on_batch(rnd_pred, y_neg)
            loss, acc = dis_trainer.train_on_batch(x_batch, y_pos)

            # Show info
            gen_loss_sum += gen_loss
            gen_acc_sum += gen_acc
            dis_loss_sum += 0.5 * dis_loss + 0.5 * loss
            dis_acc_sum += 0.5 * dis_acc + 0.5 * acc

            if e == 0 and b == 0:
                print(' {:10s} | {:8s} | {:8s} | {:8s} | {:8s} | {:8s}'.format(
                    'epoch', 'done', 'gen loss', 'gen acc', 'dis loss', 'dis acc'))

            ratio = min(100.0, 100.0 * batch_end / num_data)
            print(' epoch #{:3s} | {:6.2f} % | {:8.6f} | {:8.6f} | {:8.6f} | {:8.6f}'.format(
                '%d' % (e + 1), ratio,
                gen_loss_sum / (b + 1), gen_acc_sum / (b + 1),
                dis_loss_sum / (b + 1), dis_acc_sum / (b + 1)), end='\r')

            if (b + 1) % 200 == 0:
                save_images(gen, samples, args.output, e, batch_end)

        print('')

        # Save model
        if (e + 1) % 10 == 0:
            dis.save_weights(os.path.join(args.result, 'weights_discriminator_epoch_{:04d}.hdf5'.format(e + 1)))
            gen.save_weights(os.path.join(args.result, 'weights_generator_epoch_{:04d}.hdf5'.format(e + 1)))

        # Show current generated images
        save_images(gen, samples, args.output, e)

if __name__ == '__main__':
    main()
