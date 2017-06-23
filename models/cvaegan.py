import os
import numpy as np

import keras
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, Concatenate
from keras.layers import Activation, LeakyReLU, ELU
from keras.layers import Conv2D, UpSampling2D, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K

from .utils import set_trainable
from .cond_base import CondBaseModel

def draw_normal(args):
    z_avg, z_log_var = args
    batch_size = K.shape(z_avg)[0]
    z_dims = K.shape(z_avg)[1]
    eps = K.random_normal(shape=(batch_size, z_dims), mean=0.0, stddev=1.0)
    return z_avg + K.exp(z_log_var / 2.0) * eps

class CVAEGAN(CondBaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        num_attrs=40,
        z_dims = 128,
        enc_activation='sigmoid',
        dec_activation='sigmoid',
        dis_activation='sigmoid',
        name='cvae',
        **kwargs
    ):
        super(CVAEGAN, self).__init__(name=name, **kwargs)

        self.input_shape = input_shape
        self.num_attrs = num_attrs
        self.z_dims = z_dims
        self.enc_activation = enc_activation
        self.dec_activation = dec_activation
        self.dis_activation = dis_activation

        self.encoder = None
        self.decoder = None

        self.build_model()

    def train_on_batch(self, x_batch):
        x_image, x_attr = x_batch

        batchsize = len(x_image)
        y_pos = np.zeros(batchsize, dtype=np.int32)
        y_pos = keras.utils.to_categorical(y_pos, 2)
        y_neg = np.ones(batchsize, dtype=np.int32)
        y_neg = keras.utils.to_categorical(y_neg, 2)

        # Train CVAE
        ae_loss = self.ae_trainer.train_on_batch([x_image, x_attr], x_image)

        # Train generator
        z_rand = np.random.uniform(-1.0, 1.0, size=(batchsize, self.z_dims)).astype(np.float32)
        g_loss = self.gen_trainer.train_on_batch([z_rand, x_attr], y_pos)

        # Train discriminator
        x_fake = self.f_dec.predict([z_rand, x_attr])
        d_loss_fake = self.dis_trainer.train_on_batch(x_fake, y_neg)
        d_loss_real = self.dis_trainer.train_on_batch(x_image, y_pos)

        loss = {
            'g_loss': g_loss,
            'd_loss': (d_loss_real + d_loss_fake) * 0.5,
            'ae_loss': ae_loss
        }
        return loss

    def predict(self, z_samples):
        return self.decoder.predict(z_samples)

    def save_weights(self, out_dir, epoch, batch):
        if epoch % 10 == 0:
            self.encoder.save_weights(os.path.join(out_dir, 'enc_weights_epoch_%04d_batch_%d.hdf5' % (epoch, batch)))
            self.decoder.save_weights(os.path.join(out_dir, 'dec_weights_epoch_%04d_batch_%d.hdf5' % (epoch, batch)))

    def build_model(self):
        self.f_enc = self.build_encoder(output_dims=self.z_dims*2)
        self.f_dec = self.build_decoder()
        self.f_dis = self.build_discriminator()

        # Build discriminator
        x_data = Input(shape=self.input_shape)
        y = self.f_dis(x_data)

        self.dis_trainer = Model(x_data, y, name="Discriminator")
        self.dis_trainer.compile(loss=keras.losses.binary_crossentropy,
                                 optimizer=Adam(lr=1.0e-5, beta_1=0.5))

        self.dis_trainer.summary()

        # Build CVAE
        set_trainable(self.f_dis, False)

        x_inputs = Input(shape=self.input_shape)
        a_inputs = Input(shape=(self.num_attrs,))

        z_params = self.f_enc([x_inputs, a_inputs])
        z_avg = Lambda(lambda x: x[:, :self.z_dims], output_shape=(self.z_dims,))(z_params)
        z_log_var = Lambda(lambda x: x[:, self.z_dims:], output_shape=(self.z_dims,))(z_params)

        z = Lambda(draw_normal, output_shape=(self.z_dims,))([z_avg, z_log_var])
        x_rec = self.f_dec([z, a_inputs])

        self.ae_trainer = Model([x_inputs, a_inputs], x_rec, name='CVAE')
        self.ae_trainer.compile(loss=self.variational_loss(x_inputs, x_rec, z_avg, z_log_var),
                                optimizer=Adam(lr=2.0e-4, beta_1=0.5))

        self.ae_trainer.summary()

        # Buiild generator
        set_trainable(self.f_enc, False)

        z_rand = Input(shape=(self.z_dims,))
        a_rand = Input(shape=(self.num_attrs,))

        x_fake = self.f_dec([z_rand, a_rand])
        y_fake = self.f_dis(x_fake)
        self.gen_trainer = Model([z_rand, a_rand], y_fake, name="Generator")
        self.gen_trainer.compile(loss=keras.losses.binary_crossentropy,
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5))

        self.gen_trainer.summary()

    def build_encoder(self, output_dims):
        x_inputs = Input(shape=self.input_shape)
        a_inputs = Input(shape=(self.num_attrs,))

        a = Reshape((1, 1, self.num_attrs))(a_inputs)
        a = UpSampling2D(size=self.input_shape[:2])(a)
        x = Concatenate(axis=-1)([x_inputs, a])

        x = self.basic_encoder_layer(x, filters=64)
        x = self.basic_encoder_layer(x, filters=128)
        x = self.basic_encoder_layer(x, filters=256)
        x = self.basic_encoder_layer(x, filters=256)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation('relu')(x)

        x = Dense(output_dims)(x)
        x = Activation(self.enc_activation)(x)

        return Model([x_inputs, a_inputs], x, name='encoder')

    def build_decoder(self):
        x_inputs = Input(shape=(self.z_dims,))
        a_inputs = Input(shape=(self.num_attrs,))
        x = Concatenate()([x_inputs, a_inputs])

        x = Dense(4 * 4 * 256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((4, 4, 256))(x)

        x = self.basic_decoder_layer(x, filters=256)
        x = self.basic_decoder_layer(x, filters=128)
        x = self.basic_decoder_layer(x, filters=64)
        x = self.basic_decoder_layer(x, filters=3, activation=self.dec_activation)

        return Model([x_inputs, a_inputs], x, name='decoder')

    def build_discriminator(self):
        inputs = Input(shape=self.input_shape)

        x = self.basic_encoder_layer(inputs, filters=64)
        x = self.basic_encoder_layer(x, filters=128)
        x = self.basic_encoder_layer(x, filters=256)
        x = self.basic_encoder_layer(x, filters=256)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation('relu')(x)

        x = Dense(2)(x)
        x = Activation(self.dis_activation)(x)

        return Model(inputs, x)

    def basic_encoder_layer(self, x, filters, activation='relu'):
        x = Conv2D(filters=filters, kernel_size=(5, 5),
                   strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        if activation == 'leaky_relu':
            x = LeakyReLU(0.2)(x)
        else:
            x = Activation(activation)(x)

        return x

    def basic_decoder_layer(self, x, filters, activation='leaky_relu'):
        x = Conv2D(filters=filters, kernel_size=(5, 5), padding='same')(x)
        x = BatchNormalization()(x)
        if activation == 'leaky_relu':
            x = LeakyReLU(0.2)(x)
        else:
            x = Activation(activation)(x)

        x = UpSampling2D(size=(2, 2))(x)
        return x

    def variational_loss(self, x_true, x_pred, z_avg, z_log_var):
        def lossfun(y0, y1):
            size = K.shape(x_true)[1:]
            scale = K.cast(K.prod(size), 'float32')
            entropy = K.mean(keras.metrics.binary_crossentropy(x_true, x_pred)) * scale
            kl_loss = K.mean(-0.5 * K.sum(1.0 + z_log_var - K.square(z_avg) - K.exp(z_log_var), axis=-1))
            return entropy + kl_loss

        return lossfun
