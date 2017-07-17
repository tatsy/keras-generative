import os
import numpy as np

import keras
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, Concatenate
from keras.layers import Activation, LeakyReLU, ELU
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K

from .base import BaseModel
from .layers import *
from .utils import *

class DiscriminatorLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(DiscriminatorLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_real, y_real_rec, y_fake, y_fake_rec):
        loss_real = K.mean(K.abs(y_real - y_real_rec))
        loss_fake = K.mean(K.abs(y_fake - y_fake_rec))
        loss = loss_real - loss_fake
        return loss

    def call(self, inputs):
        y_real = inputs[0]
        y_real_rec = inputs[1]
        y_fake = inputs[2]
        y_fake_rec= inputs[3]

        loss = self.lossfun(y_real, y_real_rec, y_fake, y_fake_rec)
        self.add_loss(loss, inputs=inputs)

        return y_real

class GeneratorLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(GeneratorLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_fake, y_fake_rec):
        loss = K.mean(K.abs(y_fake - y_fake_rec))
        return loss

    def call(self, inputs):
        y_fake = inputs[0]
        y_fake_rec = inputs[1]

        loss = self.lossfun(y_fake, y_fake_rec)
        self.add_loss(loss, inputs=inputs)

        return y_fake

class EBGAN(BaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        z_dims = 128,
        name='ebgan',
        **kwargs
    ):
        super(EBGAN, self).__init__(input_shape=input_shape, name=name, **kwargs)

        self.z_dims = z_dims

        self.f_gen = None
        self.f_dis = None
        self.gen_trainer = None
        self.dis_trainer = None

        self.build_model()

    def train_on_batch(self, x_real):
        batchsize = len(x_real)
        dummy = np.zeros_like(x_real)

        z_sample = np.random.uniform(-1.0, 1.0, size=(batchsize, self.z_dims))
        z_sample = z_sample.astype('float32')

        g_loss = self.gen_trainer.train_on_batch([x_real, z_sample], dummy)
        d_loss = self.dis_trainer.train_on_batch([x_real, z_sample], dummy)

        loss = {
            'g_loss': g_loss,
            'd_loss': d_loss
        }
        return loss

    def predict(self, z_samples):
        return self.f_gen.predict(z_samples)

    def build_model(self):
        self.f_gen = self.build_decoder()
        self.f_dis = self.build_autoencoder()

        x_real = Input(shape=self.input_shape)
        z_sample = Input(shape=(self.z_dims,))

        x_fake = self.f_gen(z_sample)

        x_real_rec = self.f_dis(x_real)
        x_fake_rec = self.f_dis(x_fake)

        d_loss = DiscriminatorLossLayer()([x_real, x_real_rec, x_fake, x_fake_rec])
        g_loss = GeneratorLossLayer()([x_fake, x_fake_rec])

        # Build discriminator trainer
        set_trainable(self.f_gen, False)
        set_trainable(self.f_dis, True)

        self.dis_trainer = Model(inputs=[x_real, z_sample],
                                 outputs=[d_loss])
        self.dis_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5))
        self.dis_trainer.summary()

        # Build generator trainer
        set_trainable(self.f_gen, True)
        set_trainable(self.f_dis, False)

        self.gen_trainer = Model(inputs=[x_real, z_sample],
                                 outputs=[g_loss])
        self.gen_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5))
        self.gen_trainer.summary()

        # Store trainers
        self.store_to_save('gen_trainer')
        self.store_to_save('dis_trainer')

    def build_autoencoder(self):
        enc = self.build_encoder()
        dec = self.build_decoder()

        inputs = Input(self.input_shape)
        z = enc(inputs)
        x = dec(z)

        return Model(inputs, x)

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = BasicConvLayer(filters=128, strides=(2, 2))(inputs)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)
        x = BasicConvLayer(filters=512, strides=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation('relu')(x)

        x = Dense(self.z_dims)(x)
        x = Activation('linear')(x)

        return Model(inputs, x)

    def build_decoder(self):
        inputs = Input(shape=(self.z_dims,))
        w = self.input_shape[0] // (2 ** 3)
        x = Dense(w * w * 512)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((w, w, 512))(x)

        x = BasicDeconvLayer(filters=512, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=256, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=128, strides=(2, 2))(x)

        d = self.input_shape[2]
        x = BasicDeconvLayer(filters=d, strides=(1, 1), bnorm=False, activation='tanh')(x)

        return Model(inputs, x)
