import os
import numpy as np

import keras
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Reshape
from keras.layers import Activation, LeakyReLU, ELU
from keras.layers import Conv2D, UpSampling2D, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K

from .base import BaseModel
from .utils import *
from .layers import *

class DiscriminatorLossLayer(Layer):
    __name__ = 'discriminator_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(DiscriminatorLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_real, y_fake):
        y_pos = K.ones_like(y_real)
        y_neg = K.zeros_like(y_fake)

        loss_real = K.mean(keras.metrics.binary_crossentropy(y_pos, y_real))
        loss_fake = K.mean(keras.metrics.binary_crossentropy(y_neg, y_fake))

        return 0.5 * (loss_real + loss_fake)

    def call(self, inputs):
        y_real = inputs[0]
        y_fake = inputs[1]

        loss = self.lossfun(y_real, y_fake)
        self.add_loss(loss, inputs=inputs)

        return y_real

class GeneratorLossLayer(Layer):
    __name__ = 'generator_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(GeneratorLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_fake):
        y_pos = K.ones_like(y_fake)

        loss_fake = K.mean(keras.metrics.binary_crossentropy(y_pos, y_fake))

        return loss_fake

    def call(self, inputs):
        y_fake = inputs[0]

        loss = self.lossfun(y_fake)
        self.add_loss(loss, inputs=inputs)

        return y_fake

def discriminator_accuracy(y_real, y_fake):
    def accfun(y0, y1):
        y_pos = K.ones_like(y_real)
        y_neg = K.zeros_like(y_fake)

        acc_real = K.mean(keras.metrics.binary_accuracy(y_pos, y_real))
        acc_fake = K.mean(keras.metrics.binary_accuracy(y_neg, y_fake))

        return 0.5 * (acc_real + acc_fake)

    return accfun

def generator_accuracy(y_fake):
    def accfun(y0, y1):
        y_pos = K.ones_like(y_fake)

        acc_fake = K.mean(keras.metrics.binary_accuracy(y_pos, y_fake))

        return acc_fake

    return accfun

class DCGAN(BaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        z_dims = 128,
        name='dcgan',
        **kwargs
    ):
        super(DCGAN, self).__init__(input_shape=input_shape, name=name, **kwargs)

        self.z_dims = z_dims

        self.f_gen = None
        self.f_dis = None
        self.gen_trainer = None
        self.dis_trainer = None

        self.build_model()

    def train_on_batch(self, x_real):
        batchsize = len(x_real)
        dummy = np.zeros(batchsize, dtype='float32')

        z_sample = np.random.uniform(-1.0, 1.0, size=(batchsize, self.z_dims))
        z_sample = z_sample.astype('float32')

        g_loss, g_acc = self.gen_trainer.train_on_batch([x_real, z_sample], dummy)
        d_loss, d_acc = self.dis_trainer.train_on_batch([x_real, z_sample], dummy)

        loss = {
            'g_loss': g_loss,
            'd_loss': d_loss,
            'g_acc': g_acc,
            'd_acc': d_acc
        }
        return loss

    def predict(self, z_samples):
        return self.f_gen.predict(z_samples)

    def build_model(self):
        self.f_gen = self.build_decoder()
        self.f_dis = self.build_encoder()

        x_true = Input(shape=self.input_shape)
        z_sample = Input(shape=(self.z_dims,))

        y_pred = self.f_dis(x_true)
        x_fake = self.f_gen(z_sample)
        y_fake = self.f_dis(x_fake)

        d_loss = DiscriminatorLossLayer()([y_pred, y_fake])
        g_loss = GeneratorLossLayer()([y_fake])

        # Build discriminator trainer
        set_trainable(self.f_gen, False)
        set_trainable(self.f_dis, True)

        self.dis_trainer = Model(inputs=[x_true, z_sample],
                                 outputs=[d_loss])
        self.dis_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=1.0e-5, beta_1=0.5),
                                 metrics=[discriminator_accuracy(y_pred, y_fake)])

        self.dis_trainer.summary()

        # Build generator trainer
        set_trainable(self.f_gen, True)
        set_trainable(self.f_dis, False)

        self.gen_trainer = Model(inputs=[x_true, z_sample],
                                 outputs=[g_loss])
        self.gen_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5),
                                 metrics=[generator_accuracy(y_fake)])

        self.gen_trainer.summary()

        # Store trainers
        self.store_to_save('gen_trainer')
        self.store_to_save('dis_trainer')

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = BasicConvLayer(filters=64, strides=(2, 2))(inputs)
        x = BasicConvLayer(filters=128, strides=(2, 2))(inputs)
        x = BasicConvLayer(filters=256, strides=(2, 2))(inputs)
        x = BasicConvLayer(filters=256, strides=(2, 2))(inputs)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation('relu')(x)

        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        return Model(inputs, x)

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
