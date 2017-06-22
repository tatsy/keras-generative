import os
import math
import argparse

import numpy as np
import scipy.misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import keras
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Reshape, Lambda, Concatenate, RepeatVector
from keras.layers import BatchNormalization
from keras.layers import Activation, LeakyReLU
from keras.layers import Convolution2D, Deconvolution2D, UpSampling1D, UpSampling2D
from keras import backend as K

seed_dims = 50
num_ages = 6
image_shape = (128, 128, 3)

def normal(args):
    avg, logvar = args
    batch_size = K.shape(avg)[0]
    eps = K.random_normal(shape=(batch_size, seed_dims), mean=0.0, stddev=1.0)
    return avg + K.exp(logvar / 2.0) * eps

def progress_bar(x, maxval, width=40):
    tick = int(x / maxval * width)
    tick = min(tick, width)

    if tick == width:
        return '=' * tick

    return '=' * tick + '>' + ' ' * (width - tick - 1)

def load_data(folder, num_images=21000):
    files = [f for f in os.listdir(folder) if not f.startswith('.')]
    files = [os.path.join(folder, f) for f in files if f.endswith('.jpg')]

    if len(files) >= num_images:
        files = files[:num_images]

    print('Loading images...')
    n_images = len(files)
    x_data = [None] * n_images
    y_data = np.zeros((n_images), dtype='int32')
    for i, f in enumerate(files):
        x_data[i] = scipy.misc.imread(f, mode='RGB')
        x_data[i] = scipy.misc.imresize(x_data[i], image_shape[:2])

        base = os.path.basename(f)
        age = int(base.split('_')[0])
        y_data[i] = age

        ratio = 100.0 * (i + 1) / n_images
        print('[ {:6.2f} % ] [ {} ]'.format(ratio, progress_bar(i + 1, n_images)), end='\r', flush=True)

    x_data = np.stack(x_data, axis=0)
    print('')

    print(n_images, 'images loaded!')
    return x_data, y_data

def Encoder():
    x_inputs = Input(shape=image_shape)
    y_inputs = Input(shape=(1,))

    y = Reshape((1, 1, 1))(y_inputs)
    y = UpSampling2D(size=(image_shape[0], image_shape[1]))(y)

    x = Concatenate()([x_inputs, y])

    x = Convolution2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Convolution2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Convolution2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Convolution2D(filters=512, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Convolution2D(filters=512, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    avg = Dense(seed_dims)(x)
    log_var = Dense(seed_dims)(x)

    return Model([x_inputs, y_inputs], [avg, log_var], name='encoder')

def Generator():
    y_inputs = Input(shape=(1,))
    y = RepeatVector(seed_dims)(y_inputs)
    y = Flatten()(y)

    z_inputs = Input(shape=(seed_dims,))

    x = Concatenate()([y, z_inputs])
    x = Dense(4 * 4 * 512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Reshape((4, 4, 512))(x)

    x = Deconvolution2D(filters=512, kernel_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Deconvolution2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Deconvolution2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Deconvolution2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Deconvolution2D(filters=3, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
    x = Activation('tanh')(x)

    return Model([y_inputs, z_inputs], x, name='generator')

def Discriminator():
    inputs = Input(shape=image_shape)
    x = Convolution2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Convolution2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Convolution2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Convolution2D(filters=512, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Convolution2D(filters=512, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    x = Dense(2)(x)
    x = Activation('softmax')(x)

    model = Model(inputs, x, name='discriminator')
    return model

def variational_loss(avg, log_var):
    def lossfun(xy_true, x_pred):
        x_true = xy_true[0]
        size = K.shape(x_true)[1:]
        scale = K.cast(K.prod(size), 'float32')
        entropy = K.mean(keras.metrics.binary_crossentropy((x_true + 1.0) * 0.5, (x_pred[0] + 1.0) * 0.5)) * scale
        kl_loss = K.mean(-0.5 * K.sum(1.0 + log_var - K.square(avg) - K.exp(log_var), axis=-1))
        return entropy + kl_loss

    return lossfun

def set_trainable(model, train):
    model.trainable = train
    for l in model.layers:
        l.trainable = train

def main():
    parser = argparse.ArgumentParser(description='Keras VAE')
    parser.add_argument('--dataset', default='.', required=True)
    parser.add_argument('--output', default='output')
    parser.add_argument('--result', default='result')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=50)
    parser.add_argument('--datasize', type=int, default=20000)

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    if not os.path.exists(args.result):
        os.mkdir(args.result)

    x_data, y_data = load_data(args.dataset, args.datasize+10)
    x_data = x_data.astype('float32')
    x_data = x_data / 127.5 - 1.0
    y_data = y_data.astype('float32')
    y_data = y_data / 100.0

    x_train = x_data[:args.datasize]
    y_train = y_data[:args.datasize]
    x_test = x_data[args.datasize:args.datasize+10]
    x_test = np.concatenate([x_test] * 7)
    _, y_test = np.meshgrid(range(10), np.arange(10, 80, 10))
    y_test = y_test.flatten() / 100.0

    enc = Encoder()
    enc.summary()
    gen = Generator()
    gen.summary()
    dis = Discriminator()
    dis.summary()

    # CVAE trainer
    x_inputs = Input(shape=image_shape)
    y_inputs = Input(shape=(1,))

    avg, log_var = enc([x_inputs, y_inputs])
    z = Lambda(normal, output_shape=(seed_dims,))([avg, log_var])
    x_pred = gen([y_inputs, z])

    vae_trainer = Model([x_inputs, y_inputs], x_pred)
    vae_trainer.compile(loss=variational_loss(avg, log_var),
                        optimizer=keras.optimizers.Adam(lr=2.0e-4, beta_1=0.5))

    # Generator trainer
    set_trainable(enc, False)
    set_trainable(dis, False)
    y_rand = Input(shape=(1,))
    z_rand = Input(shape=(seed_dims,))
    x_rand = gen([y_rand, z_rand])
    l_rand = dis(x_rand)
    l_pred = dis(x_pred)
    gen_trainer = Model([x_inputs, y_inputs, y_rand, z_rand], [l_pred, l_rand])
    gen_trainer.compile(loss=keras.losses.binary_crossentropy,
                       optimizer=keras.optimizers.Adam(lr=2.0e-4, beta_1=0.5),
                       metrics=['accuracy'])

    # Discriminator trainer
    set_trainable(gen, False)
    set_trainable(dis, True)
    x_data = Input(shape=image_shape)
    l_data = dis(x_data)
    dis_trainer = Model(inputs=[x_inputs, y_inputs, y_rand, z_rand, x_data],
                        outputs=[l_pred, l_rand, l_data])
    dis_trainer.compile(loss=keras.losses.binary_crossentropy,
                        optimizer=keras.optimizers.Adam(lr=1.0e-4, beta_1=0.5),
                        metrics=['accuracy'])

    # Training loop
    num_data = len(y_train)
    for e in range(args.epoch):
        perm = np.random.permutation(num_data)
        gen_loss_sum = np.float32(0.0)
        gen_acc_sum = np.float32(0.0)
        dis_loss_sum = np.float32(0.0)
        dis_acc_sum = np.float32(0.0)

        for b in range(0, num_data, args.batchsize):
            batchsize = min(args.batchsize, num_data - b)
            indx = perm[b:b+batchsize]

            y_pos = np.ones(batchsize, np.int32)
            y_pos = keras.utils.to_categorical(y_pos, 2)
            y_neg = np.zeros(batchsize, np.int32)
            y_neg = keras.utils.to_categorical(y_neg, 2)

            y_rand_batch = np.random.random(batchsize).astype(np.float32)
            z_rand_batch = np.random.normal(size=(batchsize, seed_dims))

            x_batch = x_train[indx, :, :, :]
            y_batch = y_train[indx]

            vae_loss = vae_trainer.train_on_batch([x_batch, y_batch], x_batch)
            _, _, gen_loss, _, gen_acc = gen_trainer.train_on_batch([x_batch, y_batch, y_rand_batch, z_rand_batch], [y_pos, y_pos])
            _, dis_loss1, dis_loss2, _, dis_acc1, dis_acc2, dis_acc = dis_trainer.train_on_batch([x_batch, y_batch, y_rand_batch, z_rand_batch, x_batch], [y_neg, y_neg, y_pos])

            gen_loss_sum += gen_loss
            gen_acc_sum += gen_acc
            dis_loss_sum += 0.5 * dis_loss1 + 0.5 * dis_loss2
            dis_acc_sum += 0.5 * dis_acc1 + 0.5 * dis_acc2

            if e == 0 and b == 0:
                print(' {:10s} | {:8s} | {:12s} | {:8s} | {:8s} | {:8s} | {:8s}'.format(
                    'epoch', 'done', 'vae loss', 'gen loss', 'gen acc', 'dis loss', 'dis acc'))

            ratio = 100.0 * (b + batchsize) / num_data
            print(' {:10s} | {:6.2f} % | {:12.6f} | {:8.6f} | {:8.6f} | {:8.6f} | {:8.6f}'.format(
                'epoch #{}'.format(e + 1), ratio, vae_loss,
                gen_loss_sum / (b + 1), gen_acc_sum / (b + 1),
                dis_loss_sum / (b + 1), dis_acc_sum / (b + 1)), end='\r')

        print('')

        # Save model
        if (e + 1) % 1 == 0:
            dis.save_weights(os.path.join(args.result, 'weights_discriminator_epoch_{:04d}.hdf5'.format(e + 1)))
            gen.save_weights(os.path.join(args.result, 'weights_generator_epoch_{:04d}.hdf5'.format(e + 1)))

        # Show current generated images
        z_avg_test, _ = enc.predict([x_test, y_test])
        imgs = gen.predict([y_test, z_avg_test])
        imgs = imgs * 0.5 + 0.5
        imgs = np.clip(imgs, 0.0, 1.0)

        fig = plt.figure(figsize=(8, 8))
        grid = gridspec.GridSpec(8, 10, wspace=0.1, hspace=0.1)
        for i in range(10):
            ax = plt.Subplot(fig, grid[i])
            ax.imshow(x_test[i, :, :, :] * 0.5 + 0.5, interpolation='none', vmin=0.0, vmax=1.0)
            ax.axis('off')
            fig.add_subplot(ax)

        for i in range(10, 80):
            ax = plt.Subplot(fig, grid[i])
            ax.imshow(imgs[i-10, :, :, :], interpolation='none', vmin=0.0, vmax=1.0)
            ax.axis('off')
            fig.add_subplot(ax)

        outfile = os.path.join(args.output, 'epoch_{:04d}.png'.format(e + 1))
        fig.savefig(outfile, dpi=200)
        plt.close(fig)

if __name__ == '__main__':
    main()
