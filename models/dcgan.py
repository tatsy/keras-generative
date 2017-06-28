import os
import numpy as np

import keras
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Reshape
from keras.layers import Activation, LeakyReLU, ELU
from keras.layers import Conv2D, UpSampling2D, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K

from .utils import set_trainable
from .base import BaseModel

class DCGAN(BaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        z_dims = 128,
        enc_activation='linear',
        dec_activation='sigmoid',
        name='dcgan',
        **kwargs
    ):
        super(DCGAN, self).__init__(name=name, **kwargs)

        self.input_shape = input_shape
        self.z_dims = z_dims
        self.enc_activation = enc_activation
        self.dec_activation = dec_activation

        self.f_gen = None
        self.f_dis = None
        self.gen_trainer = None
        self.dis_trainer = None

        self.build_model()

    def train_on_batch(self, x_real):
        batchsize = len(x_real)
        y_pos = np.zeros(batchsize, dtype='float32')
        y_neg = np.ones(batchsize, dtype='float32')

        z_batch = np.random.uniform(-1.0, 1.0, size=(batchsize, self.z_dims)).astype(np.float32)

        g_loss, g_acc = self.gen_trainer.train_on_batch(z_batch, y_pos)

        x_fake = self.f_gen.predict_on_batch(z_batch)
        d_loss_real, d_acc_real = self.dis_trainer.train_on_batch(x_real, y_pos)
        d_loss_fake, d_acc_fake = self.dis_trainer.train_on_batch(x_fake, y_neg)

        loss = {
            'g_loss': g_loss,
            'd_loss': 0.5 * (d_loss_fake + d_loss_real),
            'g_acc': g_acc,
            'd_acc': d_acc
        }
        return loss

    def predict(self, z_samples):
        return self.f_gen.predict(z_samples)

    def build_model(self):
        self.f_gen = self.build_decoder()
        self.f_dis = self.build_encoder()

        dis_input = Input(shape=self.input_shape)
        y = self.f_dis(dis_input)

        self.dis_trainer = Model(dis_input, y)
        self.dis_trainer.compile(loss=keras.losses.binary_crossentropy,
                                 optimizer=Adam(lr=1.0e-5, beta_1=0.5),
                                 metrics=['accuracy'])

        set_trainable(self.f_dis, False)

        gen_input = Input(shape=(self.z_dims,))
        x = self.f_gen(gen_input)
        y_fake = self.f_dis(x)

        self.gen_trainer = Model(gen_input, y_fake)
        self.gen_trainer.compile(loss=keras.losses.binary_crossentropy,
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5),
                                 metrics=['accuracy'])

        self.gen_trainer.summary()
        self.dis_trainer.summary()

        # Store trainers
        self.store_to_save('gen_trainer')
        self.store_to_save('dis_trainer')

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = self.basic_encoder_layer(inputs, filters=64)
        x = self.basic_encoder_layer(x, filters=128)
        x = self.basic_encoder_layer(x, filters=256)
        x = self.basic_encoder_layer(x, filters=256)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation('relu')(x)

        x = Dense(1)(x)
        x = Activation(self.enc_activation)(x)

        return Model(inputs, x, name='encoder')

    def build_decoder(self):
        inputs = Input(shape=(self.z_dims,))
        x = Dense(4 * 4 * 256)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((4, 4, 256))(x)

        x = self.basic_decoder_layer(x, filters=256)
        x = self.basic_decoder_layer(x, filters=128)
        x = self.basic_decoder_layer(x, filters=64)
        x = self.basic_decoder_layer(x, filters=3, activation=self.dec_activation)

        return Model(inputs, x, name='decoder')

    def basic_encoder_layer(self, x, filters, bn=False, activation='leaky_relu'):
        x = Conv2D(filters=filters, kernel_size=(5, 5),
                   strides=(2, 2), padding='same')(x)

        if bn:
            x = BatchNormalization()(x)

        if activation == 'leaky_relu':
            x = LeakyReLU(0.2)(x)
        elif activation == 'elu':
            x = ELU()(x)
        else:
            x = Activation(activation)(x)

        return x

    def basic_decoder_layer(self, x, filters, bn=True, activation='elu'):
        x = Conv2D(filters=filters, kernel_size=(5, 5), padding='same')(x)

        if bn:
            x = BatchNormalization()(x)

        if activation == 'leaky_relu':
            x = LeakyReLU(0.2)(x)
        elif activation == 'elu':
            x = ELU()(x)
        else:
            x = Activation(activation)(x)

        x = UpSampling2D(size=(2, 2))(x)
        return x
