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

from .utils import set_trainable
from .base import BaseModel

def zero_loss(y_true, y_pred):
    return K.zeros_like(y_true)

class DiscriminatorLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(DiscriminatorLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_real, y_fake, y_real_pred, y_fake_pred):
        loss_real = K.mean(K.abs(y_real - y_real_pred))
        loss_fake = K.mean(K.abs(y_fake - y_fake_pred))
        loss = loss_real - loss_fake
        return loss

    def call(self, inputs):
        y_real = inputs[0]
        y_fake = inputs[1]
        y_real_pred = inputs[2]
        y_fake_pred = inputs[3]

        loss = self.lossfun(y_real, y_fake, y_real_pred, y_fake_pred)
        self.add_loss(loss, inputs=inputs)

        return y_real

class GeneratorLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(GeneratorLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_fake, y_fake_pred):
        loss = K.mean(K.abs(y_fake - y_fake_pred))
        return loss

    def call(self, inputs):
        y_fake = inputs[0]
        y_fake_pred = inputs[1]

        loss = self.lossfun(y_fake, y_fake_pred)
        self.add_loss(loss, inputs=inputs)

        return y_fake

class EBGAN(BaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        z_dims = 128,
        enc_activation='sigmoid',
        dec_activation='sigmoid',
        name='ebgan',
        **kwargs
    ):
        super(EBGAN, self).__init__(name=name, **kwargs)

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

        z_batch = np.random.uniform(-1.0, 1.0, size=(batchsize, self.z_dims)).astype(np.float32)

        g_loss = self.gen_trainer.train_on_batch(z_batch, x_real)

        x_fake = self.f_gen.predict_on_batch(z_batch)

        x_real_fake = np.concatenate([x_real, x_fake], axis=-1)

        d_loss = self.dis_trainer.train_on_batch(x_real_fake, x_real_fake)

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

        # Build discriminator trainer
        input_h, input_w, input_d = self.input_shape
        x_real_fake = Input(shape=(input_h, input_w, input_d * 2))

        x_real = Lambda(lambda x: x[:, :, :, :input_d], output_shape=self.input_shape)(x_real_fake)
        x_fake = Lambda(lambda x: x[:, :, :, input_d:], output_shape=self.input_shape)(x_real_fake)

        x_real_pred = self.f_dis(x_real)
        x_fake_pred = self.f_dis(x_fake)

        x_real_fake_pred = Concatenate(axis=-1)([x_real_pred, x_fake_pred])

        d_loss = DiscriminatorLossLayer()([x_real, x_fake, x_real_pred, x_fake_pred])

        self.dis_trainer = Model(x_real_fake, d_loss)
        self.dis_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=1.0e-4, beta_1=0.5))
        self.dis_trainer.summary()

        # Build generator trainer
        set_trainable(self.f_dis, False)

        z_input = Input(shape=(self.z_dims,))
        x_fake = self.f_gen(z_input)
        x_fake_pred = self.f_dis(x_fake)

        g_loss = GeneratorLossLayer()([x_fake, x_fake_pred])

        self.gen_trainer = Model(z_input, g_loss)
        self.gen_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=1.0e-4, beta_1=0.5))
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

        return Model(inputs, x, name='autoencoder')

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = self.basic_encode_layer(inputs, filters=128)
        x = self.basic_encode_layer(x, filters=256)
        x = self.basic_encode_layer(x, filters=256)
        x = self.basic_encode_layer(x, filters=512)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation('relu')(x)

        x = Dense(self.z_dims)(x)
        x = Activation(self.enc_activation)(x)

        return Model(inputs, x)

    def build_decoder(self):
        inputs = Input(shape=(self.z_dims,))
        x = Dense(4 * 4 * 512)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((4, 4, 512))(x)

        x = self.basic_decode_layer(x, filters=512)
        x = self.basic_decode_layer(x, filters=256)
        x = self.basic_decode_layer(x, filters=128)
        x = self.basic_decode_layer(x, filters=3, activation=self.dec_activation)

        return Model(inputs, x)

    def basic_encode_layer(self, x, filters, bn=False, activation='relu'):
        x = Conv2D(filters=filters, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)

        if bn:
            x = BatchNormalization()(x)

        if activation == 'leaky_relu':
            x = LeakyReLU(0.2)(x)
        elif activation == 'elu':
            x = ELU()(x)
        else:
            x = Activation(activation)(x)

        return x

    def basic_decode_layer(self, x, filters, bn=True, activation='relu'):
        x = Conv2DTranspose(filters=filters, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)

        if bn:
            x = BatchNormalization()(x)

        if activation == 'leaky_relu':
            x = LeakyReLU(0.2)(x)
        elif activation == 'elu':
            x = ELU()(x)
        else:
            x = Activation(activation)(x)

        return x
