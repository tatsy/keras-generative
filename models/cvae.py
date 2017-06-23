import os

import keras
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, Concatenate
from keras.layers import Activation, LeakyReLU, ELU
from keras.layers import Conv2D, UpSampling2D, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K

from .cond_base import CondBaseModel

def draw_normal(args):
    z_avg, z_log_var = args
    batch_size = K.shape(z_avg)[0]
    z_dims = K.shape(z_avg)[1]
    eps = K.random_normal(shape=(batch_size, z_dims), mean=0.0, stddev=1.0)
    return z_avg + K.exp(z_log_var / 2.0) * eps

class CVAE(CondBaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        num_attrs=40,
        z_dims = 128,
        enc_activation='sigmoid',
        dec_activation='sigmoid',
        name='cvae',
        **kwargs
    ):
        super(CVAE, self).__init__(name=name, **kwargs)

        self.input_shape = input_shape
        self.num_attrs = num_attrs
        self.z_dims = z_dims
        self.enc_activation = enc_activation
        self.dec_activation = dec_activation

        self.encoder = None
        self.decoder = None

        self.build_model()

    def train_on_batch(self, x_batch):
        x_image, x_attr = x_batch

        loss = {}
        loss['loss'] = self.trainer.train_on_batch([x_image, x_attr], x_image)
        return loss

    def predict(self, z_samples):
        return self.decoder.predict(z_samples)

    def save_weights(self, out_dir, epoch, batch):
        if epoch % 10 == 0:
            self.encoder.save_weights(os.path.join(out_dir, 'enc_weights_epoch_%04d_batch_%d.hdf5' % (epoch, batch)))
            self.decoder.save_weights(os.path.join(out_dir, 'dec_weights_epoch_%04d_batch_%d.hdf5' % (epoch, batch)))

    def build_model(self):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        x_inputs = Input(shape=self.input_shape)
        a_inputs = Input(shape=(self.num_attrs,))

        z_avg, z_log_var = self.encoder([x_inputs, a_inputs])
        z = Lambda(draw_normal, output_shape=(self.z_dims,))([z_avg, z_log_var])
        y = self.decoder([z, a_inputs])

        self.trainer = Model([x_inputs, a_inputs], y)
        self.trainer.compile(loss=self.variational_loss(x_inputs, y, z_avg, z_log_var),
                             optimizer=Adam(lr=2.0e-4, beta_1=0.5))

        self.trainer.summary()

    def build_encoder(self):
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

        z_avg = Dense(self.z_dims)(x)
        z_log_var = Dense(self.z_dims)(x)
        z_avg = Activation(self.enc_activation)(z_avg)
        z_log_var = Activation(self.enc_activation)(z_log_var)

        return Model([x_inputs, a_inputs], [z_avg, z_log_var], name='encoder')

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
