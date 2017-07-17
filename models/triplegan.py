import os
import numpy as np

import keras
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, Concatenate
from keras.layers import Activation, LeakyReLU, ELU
from keras.layers import Conv2D, UpSampling2D, BatchNormalization, Dropout
from keras.optimizers import Adam, Adadelta
from keras import backend as K

from .utils import set_trainable
from .cond_base import CondBaseModel

def sample_normal(args):
    z_avg, z_log_var = args
    batch_size = K.shape(z_avg)[0]
    z_dims = K.shape(z_avg)[1]
    eps = K.random_normal(shape=(batch_size, z_dims), mean=0.0, stddev=1.0)
    return z_avg + K.exp(z_log_var / 2.0) * eps

def zero_loss(y_true, y_pred):
    return K.zeros_like(y_true)

def BasicConvLayer(
    filters,
    kernel_size,
    strides=(1, 1),
    bn=False,
    dropout=0.0,
    activation='leaky_relu'):

    def fun(inputs):
        if dropout > 0.0:
            x = Dropout(dropout)(inputs)
        else:
            x = inputs

        kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        bias_init = keras.initializers.Zeros()

        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   kernel_initializer=kernel_init,
                   bias_initializer=bias_init)(x)

        if bn:
            x = BatchNormalization()(x)

        if activation == 'leaky_relu':
            x = LeakyReLU(0.02)(x)
        elif activation == 'elu':
            x = ELU()(x)
        else:
            x = Activation(activation)(x)

        return x

    return fun

def BasicDeconvLayer(
    filters,
    kernel_size,
    upsample=True,
    bn=False,
    dropout=0.0,
    activation='leaky_relu'):

    def fun(inputs):
        if dropout > 0.0:
            x = Dropout(dropout)(inputs)
        else:
            x = inputs

        kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        bias_init = keras.initializers.Zeros()

        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   kernel_initializer=kernel_init,
                   bias_initializer=bias_init,
                   padding='same')(x)

        if bn:
            x = BatchNormalization()(x)

        if activation == 'leaky_relu':
            x = LeakyReLU(0.02)(x)
        elif activation == 'elu':
            x = ELU()(x)
        else:
            x = Activation(activation)(x)

        if upsample:
            x = UpSampling2D(size=(2, 2))(x)

        return x

    return fun


class DiscriminatorLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(DiscriminatorLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_real, y_fake_x, y_fake_c):
        y_pos = K.ones_like(y_real)
        y_neg = K.zeros_like(y_real)

        loss_real = keras.metrics.binary_crossentropy(y_pos, y_real)
        loss_fake_x = keras.metrics.binary_crossentropy(y_neg, y_fake_x)
        loss_fake_c = keras.metrics.binary_crossentropy(y_neg, y_fake_c)

        return K.mean(loss_real + loss_fake_x + loss_fake_c)

    def call(self, inputs):
        y_real = inputs[0]
        y_fake_x = inputs[1]
        y_fake_c = inputs[2]
        loss = self.lossfun(y_real, y_fake_x, y_fake_c)
        self.add_loss(loss, inputs=inputs)

        return y_real

class GeneratorLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(GeneratorLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_fake, c_real, c_fake):
        y_pos = K.ones_like(y_fake)

        y_loss = keras.metrics.binary_crossentropy(y_pos, y_fake)
        c_loss = keras.metrics.binary_crossentropy(c_real, c_fake)

        return K.mean(y_loss + c_loss)

    def call(self, inputs):
        y_fake = inputs[0]
        c_real = inputs[1]
        c_fake = inputs[2]
        loss = self.lossfun(y_fake, c_real, c_fake)
        self.add_loss(loss, inputs=inputs)

        return y_fake

class ClassifierLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(ClassifierLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_fake, c_real, c_fake):
        y_pos = K.ones_like(y_fake)

        y_loss_fake = keras.metrics.binary_crossentropy(y_pos, y_fake)
        c_loss = keras.metrics.binary_crossentropy(c_real, c_fake)

        return K.mean(c_loss + y_loss_fake)

    def call(self, inputs):
        y_fake = inputs[0]
        c_real = inputs[1]
        c_fake = inputs[2]
        loss = self.lossfun(y_fake, c_real, c_fake)
        self.add_loss(loss, inputs=inputs)

        return y_fake

def discriminator_accuracy(y_real, y_fake):
    def accfun(dummy0, dummy1):
        y_pos = K.ones_like(y_real)
        y_neg = K.zeros_like(y_fake)
        acc_real = keras.metrics.binary_accuracy(y_pos, y_real)
        acc_fake = keras.metrics.binary_accuracy(y_neg, y_fake)
        return 0.5 * K.mean(acc_real + acc_fake)

    return accfun

def generator_accuracy(y_fake):
    def accfun(dummy0, dummy1):
        y_pos = K.ones_like(y_fake)
        acc_fake = keras.metrics.binary_accuracy(y_pos, y_fake)
        return K.mean(acc_fake)

    return accfun

class TripleGAN(CondBaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        num_attrs=40,
        z_dims = 128,
        name='triple_gan',
        **kwargs
    ):
        super(TripleGAN, self).__init__(name=name, **kwargs)

        self.input_shape = input_shape
        self.z_dims = z_dims
        self.num_attrs = num_attrs

        self.f_gen = None
        self.f_cls = None
        self.f_dis = None

        self.gen_trainer = None
        self.cls_trainer = None
        self.dis_trainer = None

        self.build_model()

    def train_on_batch(self, x_batch):
        x_real, c_real = x_batch

        batchsize = len(x_real)
        y_pos = np.ones(batchsize, dtype='float32')
        y_neg = np.zeros(batchsize, dtype='float32')

        z_sample = np.random.normal(size=(batchsize, self.z_dims)).astype('float32')

        g_loss, g_acc = self.gen_trainer.train_on_batch([x_real, c_real, z_sample], y_pos)
        c_loss = self.cls_trainer.train_on_batch([x_real, c_real, z_sample], y_pos)
        d_loss, d_acc = self.dis_trainer.train_on_batch([x_real, c_real, z_sample], y_pos)

        losses = {
            'g_loss': g_loss,
            'd_loss': d_loss,
            'c_loss': c_loss,
            'g_acc': g_acc,
            'd_acc': d_acc
        }
        return losses

    def predict(self, z_samples):
        return self.f_gen.predict(z_samples)

    def build_model(self):
        self.f_gen = self.build_generator()
        self.f_cls = self.build_classifier()
        self.f_dis = self.build_discriminator()

        self.f_gen.summary()
        self.f_cls.summary()
        self.f_dis.summary()

        # Build discriminator
        set_trainable(self.f_gen, False)
        set_trainable(self.f_cls, False)
        set_trainable(self.f_dis, True)

        x_real = Input(shape=self.input_shape)
        c_real = Input(shape=(self.num_attrs,))
        z_sample = Input(shape=(self.z_dims,))

        x_fake = self.f_gen([z_sample, c_real])
        c_fake = self.f_cls(x_real)

        y_real = self.f_dis([x_real, c_real])
        y_fake_x = self.f_dis([x_fake, c_real])
        y_fake_c = self.f_dis([x_real, c_fake])

        d_loss = DiscriminatorLossLayer()([y_real, y_fake_x, y_fake_c])
        self.dis_trainer = Model([x_real, c_real, z_sample], d_loss)
        self.dis_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=1.0e-5, beta_1=0.1),
                                 metrics=[discriminator_accuracy(y_real, y_fake_x)])
        self.dis_trainer.summary()

        # Build generators
        set_trainable(self.f_gen, True)
        set_trainable(self.f_cls, False)
        set_trainable(self.f_dis, False)

        c_fake_from_x_fake = self.f_cls(x_fake)

        g_loss = GeneratorLossLayer()([y_fake_x, c_real, c_fake_from_x_fake])
        self.gen_trainer = Model([x_real, c_real, z_sample], g_loss)
        self.gen_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=1.0e-4, beta_1=0.5),
                                 metrics=[generator_accuracy(y_fake_x)])
        self.gen_trainer.summary()

        # Build classifier
        set_trainable(self.f_gen, False)
        set_trainable(self.f_cls, True)
        set_trainable(self.f_dis, False)

        c_loss = ClassifierLossLayer()([y_fake_c, c_real, c_fake])
        self.cls_trainer = Model([x_real, c_real, z_sample], c_loss)
        self.cls_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=1.0e-4, beta_1=0.5))
        self.cls_trainer.summary()

        # Store trainers
        self.store_to_save('dis_trainer')
        self.store_to_save('gen_trainer')
        self.store_to_save('cls_trainer')

    def build_classifier(self):
        x_inputs = Input(shape=self.input_shape)

        x = BasicConvLayer(filters=64, kernel_size=(5, 5), bn=True)(x_inputs)
        x = BasicConvLayer(filters=128, kernel_size=(5, 5), strides=(2, 2), bn=True)(x)
        x = BasicConvLayer(filters=256, kernel_size=(5, 5), strides=(2, 2), bn=True)(x)
        x = BasicConvLayer(filters=512, kernel_size=(5, 5), strides=(2, 2), bn=True)(x)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = BatchNormalization()(x)

        x = Dense(self.num_attrs)(x)
        x = Activation('sigmoid')(x)

        return Model(x_inputs, x, name='classifier')

    def build_generator(self):
        z_inputs = Input(shape=(self.z_dims,))
        c_inputs = Input(shape=(self.num_attrs,))

        x = Concatenate(axis=-1)([z_inputs, c_inputs])

        x = Dense(4 * 4 * 512)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((4, 4, 512))(x)

        x = BasicDeconvLayer(filters=512, kernel_size=(5, 5), upsample=True, bn=True)(x)
        x = BasicDeconvLayer(filters=256, kernel_size=(5, 5), upsample=True, bn=True)(x)
        x = BasicDeconvLayer(filters=256, kernel_size=(5, 5), upsample=True, bn=True)(x)
        x = BasicDeconvLayer(filters=128, kernel_size=(5, 5), upsample=True, bn=True)(x)
        x = BasicDeconvLayer(filters=3, kernel_size=(3, 3), upsample=False, activation='tanh')(x)

        return Model([z_inputs, c_inputs], x, name='generator')

    def build_discriminator(self):
        x_inputs = Input(shape=self.input_shape)
        c_inputs = Input(shape=(self.num_attrs,))

        x = BasicConvLayer(filters=64, kernel_size=(5, 5), bn=True)(x_inputs)
        x = BasicConvLayer(filters=128, kernel_size=(5, 5), strides=(2, 2), bn=True)(x)
        x = BasicConvLayer(filters=256, kernel_size=(5, 5), strides=(2, 2), bn=True)(x)
        x = BasicConvLayer(filters=512, kernel_size=(5, 5), strides=(2, 2), bn=True)(x)

        x = Flatten()(x)
        x = Concatenate(axis=-1)([x, c_inputs])

        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        return Model([x_inputs, c_inputs], x, name='discriminator')
