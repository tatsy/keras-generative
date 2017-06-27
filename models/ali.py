import os
import numpy as np

import keras
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, Concatenate
from keras.layers import Activation, LeakyReLU, ELU
from keras.layers import Conv2D, Deconv2D, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras import backend as K

from .utils import set_trainable
from .base import BaseModel

class BasicConvLayer(Layer):
    def __init__(self,
        conv_type,
        filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        bn=False,
        dropout=0.0,
        activation='leaky_relu',
        **kwargs
    ):
        super(BasicConvLayer, self).__init__(**kwargs)
        self.conv_type = conv_type
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.bn = bn
        self.dropout = dropout
        self.activation = activation

    def build(self, input_shape):
        if self.conv_type == 'conv':
            self.conv = Conv2D(filters=self.filters,
                               kernel_size=self.kernel_size,
                               strides=self.strides)
        elif self.conv_type == 'deconv':
            self.conv = Deconv2D(filters=self.filters,
                                 kernel_size=self.kernel_size,
                                 strides=self.strides)
        else:
            raise Exception(self.conv_type, 'is not supported!')

        self.trainable_weights = self.conv.trainable_weights

    def call(self, x, mask=None):
        x = self.conv(x)

        if self.bn:
            x = BatchNormalization()(x)

        if self.activation == 'leaky_relu':
            x = LeakyReLU(0.02)(x)
        elif self.activation == 'elu':
            x = ELU()(x)
        else:
            x = Activation(self.activation)(x)

        if self.dropout > 0.0:
            x = Dropout(self.dropout)(x)

        return x

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)

    def get_output_shape_for(self, input_shape):
        return self.conv.get_output_shape_for(input_shape)

class ALI(BaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        z_dims = 128,
        enc_activation='sigmoid',
        dec_activation='sigmoid',
        name='ali',
        **kwargs
    ):
        super(ALI, self).__init__(name=name, **kwargs)

        self.input_shape = input_shape
        self.z_dims = z_dims
        self.enc_activation = enc_activation
        self.dec_activation = dec_activation

        self.f_Gz = None
        self.f_Gx = None
        self.f_D = None

        self.gen_x_trainer = None
        self.gen_z_trainer = None
        self.dis_trainer = None

        self.build_model()

    def train_on_batch(self, x_real):
        batchsize = len(x_real)
        y_pos = np.zeros(batchsize, dtype=np.int32)
        y_pos = keras.utils.to_categorical(y_pos, 2)
        y_neg = np.ones(batchsize, dtype=np.int32)
        y_neg = keras.utils.to_categorical(y_neg, 2)

        z_real = np.random.normal(size=(batchsize, self.z_dims)).astype('float32')

        g_x_loss, g_x_acc = self.gen_x_trainer.train_on_batch([x_real, z_real], y_neg)
        g_z_loss, g_z_acc = self.gen_z_trainer.train_on_batch([x_real, z_real], y_pos)
        g_loss = g_x_loss + g_z_loss
        g_acc = 0.5 * (g_x_acc + g_z_acc)

        x_fake = self.f_Gx.predict_on_batch(z_real)
        z_fake = self.f_Gz.predict_on_batch(x_real)

        d_x_loss, d_x_acc = self.dis_trainer.train_on_batch([x_real, z_fake], y_pos)
        d_z_loss, d_z_acc = self.dis_trainer.train_on_batch([x_fake, z_real], y_neg)
        d_loss = d_x_loss + d_z_loss
        d_acc = 0.5 * (d_x_acc + d_z_acc)

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

        # Build discriminator
        x_inputs = Input(shape=self.input_shape)
        z_inputs = Input(shape=(self.z_dims,))
        y_output = self.f_D([x_inputs, z_inputs])
        self.dis_trainer = Model([x_inputs, z_inputs], y_output)
        self.dis_trainer.compile(loss=keras.losses.binary_crossentropy,
                                 optimizer=Adam(lr=1.0e-5, beta_1=0.5),
                                 metrics=['accuracy'])
        self.dis_trainer.summary()

        # Build generators
        set_trainable(self.f_D, False)

        x_real = Input(shape=self.input_shape)
        z_real = Input(shape=(self.z_dims,))

        z_fake = self.f_Gz(x_real)
        x_fake = self.f_Gx(z_real)

        y_x_real_z_fake = self.f_D([x_real, z_fake])
        y_x_fake_z_real = self.f_D([x_fake, z_real])

        self.gen_x_trainer = Model([x_real, z_real], y_x_real_z_fake)
        self.gen_x_trainer.compile(loss=keras.losses.binary_crossentropy,
                                   optimizer=Adam(lr=1.0e-4, beta_1=0.5),
                                   metrics=['accuracy'])
        self.gen_x_trainer.summary()

        self.gen_z_trainer = Model([x_real, z_real], y_x_fake_z_real)
        self.gen_z_trainer.compile(loss=keras.losses.binary_crossentropy,
                                   optimizer=Adam(lr=1.0e-4, beta_1=0.5),
                                   metrics=['accuracy'])
        self.gen_z_trainer.summary()

        # Store trainers
        self.store_to_save('dis_trainer')
        self.store_to_save('gen_x_trainer')
        self.store_to_save('gen_z_trainer')

    def build_Gz(self):
        inputs = Input(shape=self.input_shape)

        x = BasicConvLayer(conv_type='conv', filters=64, kernel_size=(2, 2), bn=True)(inputs)
        x = BasicConvLayer(conv_type='conv', filters=128, kernel_size=(7, 7), strides=(2, 2), bn=True)(x)
        x = BasicConvLayer(conv_type='conv', filters=256, kernel_size=(5, 5), strides=(2, 2), bn=True)(x)
        x = BasicConvLayer(conv_type='conv', filters=256, kernel_size=(7, 7), strides=(2, 2), bn=True)(x)
        x = BasicConvLayer(conv_type='conv', filters=512, kernel_size=(4, 4), bn=True)(x)
        x = BasicConvLayer(conv_type='conv', filters=self.z_dims, kernel_size=(1, 1), activation='linear')(x)

        x = Flatten()(x)

        return Model(inputs, x, name='Gz')

    def build_Gx(self):
        inputs = Input(shape=(self.z_dims,))

        x = Reshape((1, 1, self.z_dims))(inputs)
        x = BasicConvLayer(conv_type='deconv', filters=512, kernel_size=(4, 4), bn=True)(x)
        x = BasicConvLayer(conv_type='deconv', filters=256, kernel_size=(7, 7), strides=(2, 2), bn=True)(x)
        x = BasicConvLayer(conv_type='deconv', filters=256, kernel_size=(5, 5), strides=(2, 2), bn=True)(x)
        x = BasicConvLayer(conv_type='deconv', filters=128, kernel_size=(7, 7), strides=(2, 2), bn=True)(x)
        x = BasicConvLayer(conv_type='deconv', filters=64, kernel_size=(2, 2), bn=True)(x)
        x = BasicConvLayer(conv_type='conv', filters=3, kernel_size=(1, 1), activation='sigmoid')(x)

        return Model(inputs, x)

    def build_D(self):
        x_inputs = Input(shape=self.input_shape)

        x = BasicConvLayer(conv_type='conv', filters=64, kernel_size=(2, 2), bn=True)(x_inputs)
        x = BasicConvLayer(conv_type='conv', filters=128, kernel_size=(7, 7), strides=(2, 2), bn=True)(x)
        x = BasicConvLayer(conv_type='conv', filters=256, kernel_size=(5, 5), strides=(2, 2), bn=True)(x)
        x = BasicConvLayer(conv_type='conv', filters=256, kernel_size=(7, 7), strides=(2, 2), bn=True)(x)
        x = BasicConvLayer(conv_type='conv', filters=256, kernel_size=(4, 4), bn=True)(x)

        z_inputs = Input(shape=(self.z_dims,))
        z = Reshape((1, 1, self.z_dims))(z_inputs)
        z = BasicConvLayer(conv_type='conv', filters=1024, kernel_size=(1, 1), dropout=0.2)(z)
        z = BasicConvLayer(conv_type='conv', filters=1024, kernel_size=(1, 1), dropout=0.2)(z)

        xz = Concatenate(axis=-1)([x, z])
        xz = BasicConvLayer(conv_type='conv', filters=2048, kernel_size=(1, 1), dropout=0.2)(xz)
        xz = BasicConvLayer(conv_type='conv', filters=2048, kernel_size=(1, 1), dropout=0.2)(xz)
        xz = BasicConvLayer(conv_type='conv', filters=2, kernel_size=(1, 1), dropout=0.2, activation='sigmoid')(xz)

        xz = Flatten()(xz)

        return Model([x_inputs, z_inputs], xz)
