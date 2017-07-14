import os
import numpy as np

import keras
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, Concatenate
from keras.layers import Activation, LeakyReLU, ELU
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, Dropout
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
                   bias_initializer=bias_init,
                   use_bias=True)(x)

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

        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            kernel_initializer=kernel_init,
                            bias_initializer=bias_init,
                            use_bias=True)(x)

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


class DiscriminatorLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(DiscriminatorLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_real, y_fake):
        y_pos = K.ones_like(y_real)
        # y_neg = K.zeros_like(y_real)
        loss_real = K.mean(keras.metrics.binary_crossentropy(y_pos, y_real))
        loss_fake = K.mean(keras.metrics.binary_crossentropy(y_pos, y_fake))

        return loss_real - loss_fake

    def call(self, inputs):
        y_real = inputs[0]
        y_fake = inputs[1]
        loss = self.lossfun(y_real, y_fake)
        self.add_loss(loss, inputs=inputs)

        return y_real

class GeneratorLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(GeneratorLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_real, y_fake):
        y_pos = K.ones_like(y_real)
        y_neg = K.zeros_like(y_fake)

        loss_fake = keras.metrics.binary_crossentropy(y_pos, y_fake)
        loss_real = keras.metrics.binary_crossentropy(y_neg, y_real)

        return K.mean(loss_real + loss_fake)

    def call(self, inputs):
        y_real = inputs[0]
        y_fake = inputs[1]
        loss = self.lossfun(y_real, y_fake)
        self.add_loss(loss, inputs=inputs)

        return y_real

def discriminator_accuracy(y_real, y_fake):
    y_pos = K.ones_like(y_real)
    y_neg = K.zeros_like(y_fake)
    acc_real = keras.metrics.binary_accuracy(y_pos, y_real)
    acc_fake = keras.metrics.binary_accuracy(y_neg, y_fake)
    return 0.5 * K.mean(acc_real + acc_fake)

def generator_accuracy(y_real, y_fake):
    y_pos = K.ones_like(y_fake)
    y_neg = K.zeros_like(y_real)
    acc_fake = keras.metrics.binary_accuracy(y_pos, y_fake)
    acc_real = keras.metrics.binary_accuracy(y_neg, y_real)
    return 0.5 * K.mean(acc_real + acc_fake)

class CALI(CondBaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        num_attrs=40,
        z_dims = 128,
        name='cali',
        **kwargs
    ):
        super(CALI, self).__init__(name=name, **kwargs)

        self.input_shape = input_shape
        self.z_dims = z_dims
        self.num_attrs = num_attrs

        self.f_Gz = None
        self.f_Gx = None
        self.f_D = None

        self.gen_trainer = None
        self.dis_trainer = None

        self.build_model()

    def train_on_batch(self, x_batch):
        x_real, c_real = x_batch

        batchsize = len(x_real)
        y_pos = np.ones(batchsize, dtype='float32')
        y_neg = np.zeros(batchsize, dtype='float32')

        z_fake = np.random.normal(size=(batchsize, self.z_dims)).astype('float32')

        g_loss, g_acc = self.gen_trainer.train_on_batch([x_real, c_real, z_fake], y_pos)
        d_loss, d_acc = self.dis_trainer.train_on_batch([x_real, c_real, z_fake], y_pos)

        losses = {
            'g_loss': g_loss,
            'd_loss': d_loss,
            'g_acc': g_acc,
            'd_acc': d_acc
        }
        return losses

    def predict(self, z_samples):
        return self.f_Gx.predict(z_samples)

    def build_model(self):
        self.f_Gz = self.build_Gz()
        self.f_Gx = self.build_Gx()
        self.f_D = self.build_D()

        self.f_Gz.summary()
        self.f_Gx.summary()
        self.f_D.summary()

        # Build discriminator
        set_trainable(self.f_Gz, False)
        set_trainable(self.f_Gx, False)
        set_trainable(self.f_D, True)

        x_real = Input(shape=self.input_shape)
        c_real = Input(shape=(self.num_attrs,))
        z_fake = Input(shape=(self.z_dims,))

        x_fake = self.f_Gx([z_fake, c_real])
        z_params = self.f_Gz([x_real, c_real])

        z_avg = Lambda(lambda x: x[:, :self.z_dims], output_shape=(self.z_dims,))(z_params)
        z_log_var = Lambda(lambda x: x[:, self.z_dims:], output_shape=(self.z_dims,))(z_params)
        z_real = Lambda(sample_normal, output_shape=(self.z_dims,))([z_avg, z_log_var])

        y_real = self.f_D([x_real, c_real, z_real])
        y_fake = self.f_D([x_fake, c_real, z_fake])

        d_loss = DiscriminatorLossLayer()([y_real, y_fake])
        self.dis_trainer = Model([x_real, c_real, z_fake], d_loss)
        self.dis_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=1.0e-4, beta_1=0.5),
                                 metrics=[discriminator_accuracy])
        self.dis_trainer.summary()

        # Build generators
        set_trainable(self.f_Gz, True)
        set_trainable(self.f_Gx, True)
        set_trainable(self.f_D, False)

        g_loss = GeneratorLossLayer()([y_real, y_fake])
        self.gen_trainer = Model([x_real, c_real, z_fake], g_loss)
        self.gen_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=1.0e-4, beta_1=0.5),
                                 metrics=[generator_accuracy])
        self.gen_trainer.summary()

        # Store trainers
        self.store_to_save('dis_trainer')
        self.store_to_save('gen_trainer')

    def build_Gz(self):
        x_inputs = Input(shape=self.input_shape)
        c_inputs = Input(shape=(self.num_attrs,))

        x = BasicConvLayer(filters=64, kernel_size=(2, 2), bn=True)(x_inputs)
        x = BasicConvLayer(filters=128, kernel_size=(7, 7), strides=(2, 2), bn=True)(x)
        x = BasicConvLayer(filters=256, kernel_size=(5, 5), strides=(2, 2), bn=True)(x)
        x = BasicConvLayer(filters=256, kernel_size=(7, 7), strides=(2, 2), bn=True)(x)
        x = BasicConvLayer(filters=512, kernel_size=(4, 4), bn=True)(x)

        c = Reshape((1, 1, self.num_attrs))(c_inputs)
        x = Concatenate(axis=-1)([x, c])
        x = BasicConvLayer(filters=self.z_dims * 2, kernel_size=(1, 1), activation='linear')(x)

        x = Flatten()(x)

        return Model([x_inputs, c_inputs], x, name='Gz')

    def build_Gx(self):
        x_inputs = Input(shape=(self.z_dims,))
        c_inputs = Input(shape=(self.num_attrs,))

        x = Concatenate(axis=-1)([x_inputs, c_inputs])
        x = Reshape((1, 1, self.z_dims + self.num_attrs))(x)

        x = BasicDeconvLayer(filters=512, kernel_size=(4, 4), bn=True)(x)
        x = BasicDeconvLayer(filters=256, kernel_size=(7, 7), strides=(2, 2), bn=True)(x)
        x = BasicDeconvLayer(filters=256, kernel_size=(5, 5), strides=(2, 2), bn=True)(x)
        x = BasicDeconvLayer(filters=128, kernel_size=(7, 7), strides=(2, 2), bn=True)(x)
        x = BasicDeconvLayer(filters=64, kernel_size=(2, 2), bn=True)(x)
        x = BasicDeconvLayer(filters=3, kernel_size=(1, 1), activation='tanh')(x)

        return Model([x_inputs, c_inputs], x)

    def build_D(self):
        x_inputs = Input(shape=self.input_shape)
        c_inputs = Input(shape=(self.num_attrs,))

        x = BasicConvLayer(filters=64, kernel_size=(2, 2), bn=True)(x_inputs)
        x = BasicConvLayer(filters=128, kernel_size=(7, 7), strides=(2, 2), bn=True)(x)
        x = BasicConvLayer(filters=256, kernel_size=(5, 5), strides=(2, 2), bn=True)(x)
        x = BasicConvLayer(filters=256, kernel_size=(7, 7), strides=(2, 2), bn=True)(x)
        x = BasicConvLayer(filters=512, kernel_size=(4, 4), bn=True)(x)

        z_inputs = Input(shape=(self.z_dims,))
        z = Reshape((1, 1, self.z_dims))(z_inputs)
        z = BasicConvLayer(filters=1024, kernel_size=(1, 1), dropout=0.2)(z)
        z = BasicConvLayer(filters=1024, kernel_size=(1, 1), dropout=0.2)(z)

        c = Reshape((1, 1, self.num_attrs))(c_inputs)

        xz = Concatenate(axis=-1)([x, c, z])
        xz = BasicConvLayer(filters=2048, kernel_size=(1, 1), dropout=0.2)(xz)
        xz = BasicConvLayer(filters=2048, kernel_size=(1, 1), dropout=0.2)(xz)
        xz = BasicConvLayer(filters=1, kernel_size=(1, 1), dropout=0.2, activation='sigmoid')(xz)

        xz = Flatten()(xz)

        return Model([x_inputs, c_inputs, z_inputs], xz)
