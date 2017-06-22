import numpy as np

import keras
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, Concatenate
from keras.layers import Activation, LeakyReLU, ELU
from keras.layers import Conv2D, UpSampling2D, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K

from .utils import set_trainable
from .base import BaseModel

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

    def save_weights(self, out_dir, epoch, batch):
        if epoch % 10 == 0:
            self.f_dis.save_weights(os.path.join(args.result, 'gen_weights_epoch_{:04d}.hdf5'.format(epoch)))
            self.f_gen.save_weights(os.path.join(args.result, 'dis_weights_epoch_{:04d}.hdf5'.format(epoch)))

    def build_model(self):
        self.f_gen = self.build_decoder()
        self.f_dis = self.build_autoencoder()

        input_h, input_w, input_d = self.input_shape
        x_real_fake = Input(shape=(input_h, input_w, input_d * 2))

        x_real = Lambda(lambda x: x[:, :, :, :input_d], output_shape=self.input_shape)(x_real_fake)
        x_fake = Lambda(lambda x: x[:, :, :, input_d:], output_shape=self.input_shape)(x_real_fake)

        x_real_pred = self.f_dis(x_real)
        x_fake_pred = self.f_dis(x_fake)

        x_real_fake_pred = Concatenate(axis=-1)([x_real_pred, x_fake_pred])

        self.dis_trainer = Model(x_real_fake, x_real_fake_pred)
        self.dis_trainer.compile(loss=self.discriminator_loss(x_real, x_fake, x_real_pred, x_fake_pred),
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5))

        set_trainable(self.f_dis, False)

        z_input = Input(shape=(self.z_dims,))
        x_fake = self.f_gen(z_input)
        x_fake_pred = self.f_dis(x_fake)

        self.gen_trainer = Model(z_input, x_fake_pred)
        self.gen_trainer.compile(loss=self.generator_loss(x_fake, x_fake_pred),
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5))

        self.gen_trainer.summary()
        self.dis_trainer.summary()

    def build_autoencoder(self):
        enc = self.build_encoder()
        dec = self.build_decoder()

        inputs = Input(self.input_shape)
        z = enc(inputs)
        x = dec(z)

        return Model(inputs, x, name='autoencoder')

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = self.basic_encoder_layer(inputs, filters=64)
        x = self.basic_encoder_layer(x, filters=128)
        x = self.basic_encoder_layer(x, filters=256)
        x = self.basic_encoder_layer(x, filters=256)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation('relu')(x)

        x = Dense(self.z_dims)(x)
        x = Activation(self.enc_activation)(x)

        return Model(inputs, x)

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

        return Model(inputs, x)

    def basic_encoder_layer(self, x, filters, activation='leaky_relu'):
        x = Conv2D(filters=filters, kernel_size=(3, 3),
                   strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        if activation == 'leaky_relu':
            x = LeakyReLU(0.2)(x)
        else:
            x = Activation(activation)(x)

        return x

    def basic_decoder_layer(self, x, filters, activation='relu'):
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        if activation == 'elu':
            x = ELU()(x)
        else:
            x = Activation(activation)(x)

        x = UpSampling2D(size=(2, 2))(x)
        return x

    def discriminator_loss(self, y_real, y_fake, y_real_pred, y_fake_pred):
        def losses(y0, y1):
            loss_real = K.mean(K.square(y_real - y_real_pred), axis=[1, 2, 3])
            loss_fake = K.mean(K.square(y_fake - y_fake_pred), axis=[1, 2, 3])
            return loss_real - loss_fake

        return losses

    def generator_loss(self, y_fake, y_fake_pred):
        def losses(y0, y1):
            loss = K.mean(K.square(y_fake - y_fake_pred), axis=[1, 2, 3])
            return loss

        return losses
