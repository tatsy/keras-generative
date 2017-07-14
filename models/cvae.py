import os
import numpy as np

import keras
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, Concatenate
from keras.layers import Activation, LeakyReLU, ELU
from keras.layers import Conv2D, UpSampling2D, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K

from .cond_base import CondBaseModel
from .layers import *
from .utils import *

class CVAE(CondBaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        num_attrs=40,
        z_dims = 128,
        name='cvae',
        **kwargs
    ):
        super(CVAE, self).__init__(input_shape=input_shape, name=name, **kwargs)

        self.num_attrs = num_attrs
        self.z_dims = z_dims

        self.f_gen = None
        self.f_dis = None
        self.vae_trainer = None

        self.build_model()

    def train_on_batch(self, batch):
        x_real, c_real = batch
        dummy = np.zeros_like(x_real)

        loss = self.vae_trainer.train_on_batch([x_real, c_real], dummy)
        return { 'loss': loss }

    def predict(self, z_samples):
        return self.decoder.predict(z_samples)

    def build_model(self):
        self.f_enc = self.build_encoder()
        self.f_dec = self.build_decoder()

        x_real = Input(shape=self.input_shape)
        c_real = Input(shape=(self.num_attrs,))

        z_avg, z_log_var = self.f_enc([x_real, c_real])
        z = SampleNormal()([z_avg, z_log_var])
        x_fake = self.f_dec([z, c_real])

        vae_loss = VAELossLayer()([x_real, x_fake, z_avg, z_log_var])

        self.vae_trainer = Model(inputs=[x_real, c_real],
                                 outputs=[vae_loss])
        self.vae_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5))

        self.vae_trainer.summary()

        # Store trainers
        self.store_to_save('vae_trainer')

    def build_encoder(self):
        x_inputs = Input(shape=self.input_shape)
        c_inputs = Input(shape=(self.num_attrs,))

        c = Reshape((1, 1, self.num_attrs))(c_inputs)
        c = UpSampling2D(size=self.input_shape[:2])(c)
        x = Concatenate(axis=-1)([x_inputs, c])

        x = BasicConvLayer(filters=64, strides=(2, 2))(x)
        x = BasicConvLayer(filters=128, strides=(2, 2))(x)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation('relu')(x)

        z_avg = Dense(self.z_dims)(x)
        z_log_var = Dense(self.z_dims)(x)
        z_avg = Activation('linear')(z_avg)
        z_log_var = Activation('linear')(z_log_var)

        return Model(inputs=[x_inputs, c_inputs],
                     outputs=[z_avg, z_log_var])

    def build_decoder(self):
        z_inputs = Input(shape=(self.z_dims,))
        c_inputs = Input(shape=(self.num_attrs,))
        z = Concatenate()([z_inputs, c_inputs])

        w = self.input_shape[0] // (2 ** 3)
        x = Dense(w * w * 256)(z)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((w, w, 256))(x)

        x = BasicDeconvLayer(filters=256, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=128, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=64, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=3, strides=(1, 1), bnorm=False, activation='tanh')(x)

        return Model([z_inputs, c_inputs], x)
