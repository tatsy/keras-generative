import os

import keras
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Reshape
from keras.layers import Activation, LeakyReLU, ELU
from keras.layers import Conv2D, UpSampling2D, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K

from .base import BaseModel
from .utils import *
from .layers import *

class VAE(BaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        z_dims = 128,
        name='vae',
        **kwargs
    ):
        super(VAE, self).__init__(input_shape=input_shape, name=name, **kwargs)

        self.z_dims = z_dims

        self.f_enc = None
        self.f_dec = None
        self.vae_trainer = None

        self.build_model()

    def train_on_batch(self, x_batch):
        loss = self.vae_trainer.train_on_batch(x_batch, x_batch)
        return { 'loss': loss }

    def predict(self, z_samples):
        return self.f_dec.predict(z_samples)

    def build_model(self):
        self.f_enc = self.build_encoder()
        self.f_dec = self.build_decoder()

        x_true = Input(shape=self.input_shape)
        z_avg, z_log_var = self.f_enc(x_true)
        z = SampleNormal()([z_avg, z_log_var])
        x_pred = self.f_dec(z)
        vae_loss = VAELossLayer()([x_true, x_pred, z_avg, z_log_var])

        self.vae_trainer = Model(inputs=[x_true],
                             outputs=[vae_loss])
        self.vae_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5))

        self.vae_trainer.summary()

        # Store trainers
        self.store_to_save('vae_trainer')

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = BasicConvLayer(filters=64, strides=(2, 2))(inputs)
        x = BasicConvLayer(filters=128, strides=(2, 2))(inputs)
        x = BasicConvLayer(filters=256, strides=(2, 2))(inputs)
        x = BasicConvLayer(filters=256, strides=(2, 2))(inputs)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation('relu')(x)

        z_avg = Dense(self.z_dims)(x)
        z_log_var = Dense(self.z_dims)(x)

        z_avg = Activation('linear')(z_avg)
        z_log_var = Activation('linear')(z_log_var)

        return Model(inputs, [z_avg, z_log_var])

    def build_decoder(self):
        inputs = Input(shape=(self.z_dims,))
        w = self.input_shape[0] // (2 ** 3)
        x = Dense(w * w * 256)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((w, w, 256))(x)

        x = BasicDeconvLayer(filters=256, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=128, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=64, strides=(2, 2))(x)

        d = self.input_shape[2]
        x = BasicDeconvLayer(filters=d, strides=(1, 1), bnorm=False, activation='tanh')(x)

        return Model(inputs, x)
