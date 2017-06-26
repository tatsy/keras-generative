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
        name='cvaegan',
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
        z_rand = np.random.normal(size=(batchsize, self.z_dims)).astype('float32')

        # Train autoencoder
        ae_loss = self.ae_trainer.train_on_batch([x_image, x_attr], x_image)

        # Train generator
        g_loss = self.gen_trainer.train_on_batch([x_image, z_rand, x_attr], x_image)

        # Generate fake sample
        x_fake = self.f_dec.predict_on_batch([z_rand, x_attr])

        # Train classifier
        c_loss = self.cls_trainer.train_on_batch([x_image, x_fake], x_attr)

        # Train discriminator
        y_pos = np.ones((batchsize, self.z_dims), dtype='float32')
        d_loss = self.dis_trainer.train_on_batch([x_image, x_fake], y_pos)

        loss = {
            'g_loss': g_loss,
            'd_loss': d_loss,
            'c_loss': c_loss,
            'ae_loss': ae_loss
        }
        return loss

    def predict(self, z_samples):
        return self.f_dec.predict(z_samples)

    def build_model(self):
        self.f_enc = self.build_encoder(output_dims=self.z_dims*2)
        self.f_dec = self.build_decoder()
        self.f_dis = self.build_discriminator()
        self.f_cls = self.build_classifier()

        # Build classfier trainer
        x_real = Input(shape=self.input_shape)
        x_fake = Input(shape=self.input_shape)
        c_pred_real = self.f_cls(x_real)
        c_pred_fake = self.f_cls(x_fake)

        self.cls_trainer = Model(inputs=[x_real, x_fake],
                                 outputs=[c_pred_real])
        self.cls_trainer.compile(loss=self.classifier_loss(c_pred_real, c_pred_fake),
                                 optimizer=Adam(lr=1.0e-5, beta_1=0.5))

        self.cls_trainer.summary()

        # Build discriminator trainer
        y_pred_real = self.f_dis(x_real)
        y_pred_fake = self.f_dis(x_fake)
        self.dis_trainer = Model(inputs=[x_real, x_fake],
                                 outputs=[y_pred_real])
        self.dis_trainer.compile(loss=self.discriminator_loss(y_pred_real, y_pred_fake),
                                 optimizer=Adam(lr=1.0e-5, beta_1=0.5))

        self.dis_trainer.summary()

        # Build generator trainer
        set_trainable(self.f_cls, False)
        set_trainable(self.f_dis, False)

        x_inputs = Input(shape=self.input_shape)
        c_inputs = Input(shape=(self.num_attrs,))
        z_inputs = Input(shape=(self.z_dims,))

        z_params = self.f_enc([x_inputs, c_inputs])
        z_avg = Lambda(lambda x: x[:, :self.z_dims], output_shape=(self.z_dims,))(z_params)
        z_log_var = Lambda(lambda x: x[:, self.z_dims:], output_shape=(self.z_dims,))(z_params)

        z_from_x = Lambda(draw_normal, output_shape=(self.z_dims,))([z_avg, z_log_var])
        x_rec_from_x = self.f_dec([z_from_x, c_inputs])
        x_rec_from_z = self.f_dec([z_inputs, c_inputs])

        y_pred_from_x = self.f_dis(x_rec_from_x)
        y_pred_from_z = self.f_dis(x_rec_from_z)
        y_pred_from_data = self.f_dis(x_inputs)

        c_pred_from_x = self.f_cls(x_rec_from_x)
        c_pred_from_z = self.f_cls(x_rec_from_z)
        c_pred_from_data = self.f_cls(x_inputs)

        self.gen_trainer = Model(inputs=[x_inputs, z_inputs, c_inputs],
                                 outputs=x_rec_from_x)
        self.gen_trainer.compile(
            loss=self.generator_loss(x_inputs, x_rec_from_x,
                                     y_pred_from_data, y_pred_from_x, y_pred_from_z,
                                     c_pred_from_data, c_pred_from_x, c_pred_from_z),
            optimizer=Adam(lr=1.0e-4, beta_1=0.5))

        # Build autoencoder
        set_trainable(self.f_dec, False)
        self.ae_trainer = Model(inputs=[x_inputs, c_inputs],
                                outputs=[x_rec_from_x])
        self.ae_trainer.compile(loss=self.autoencoder_loss(x_inputs, x_rec_from_x, z_avg, z_log_var),
                                optimizer=Adam(lr=1.0e-4, beta_1=0.5))

        self.ae_trainer.summary()

        # Store trainers
        self.store_to_save('cls_trainer')
        self.store_to_save('dis_trainer')
        self.store_to_save('gen_trainer')
        self.store_to_save('ae_trainer')

    def build_encoder(self, output_dims):
        x_inputs = Input(shape=self.input_shape)
        a_inputs = Input(shape=(self.num_attrs,))

        a = Reshape((1, 1, self.num_attrs))(a_inputs)
        a = UpSampling2D(size=self.input_shape[:2])(a)
        x = Concatenate(axis=-1)([x_inputs, a])

        x = self.basic_encode_layer(x, filters=64)
        x = self.basic_encode_layer(x, filters=128)
        x = self.basic_encode_layer(x, filters=256)
        x = self.basic_encode_layer(x, filters=256)

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

        x = self.basic_decode_layer(x, filters=256)
        x = self.basic_decode_layer(x, filters=128)
        x = self.basic_decode_layer(x, filters=64)
        x = self.basic_decode_layer(x, filters=3, activation=self.dec_activation)

        return Model([x_inputs, a_inputs], x, name='decoder')

    def build_discriminator(self):
        inputs = Input(shape=self.input_shape)

        x = self.basic_encode_layer(inputs, filters=64)
        x = self.basic_encode_layer(x, filters=128)
        x = self.basic_encode_layer(x, filters=256)
        x = self.basic_encode_layer(x, filters=256)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation('relu')(x)

        x = Dense(1)(x)
        x = Activation(self.dis_activation)(x)

        return Model(inputs, x)

    def build_classifier(self):
        inputs = Input(shape=self.input_shape)

        x = self.basic_encode_layer(inputs, filters=64)
        x = self.basic_encode_layer(x, filters=128)
        x = self.basic_encode_layer(x, filters=256)
        x = self.basic_encode_layer(x, filters=256)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation('relu')(x)

        x = Dense(self.num_attrs)(x)
        x = Activation(self.dis_activation)(x)

        return Model(inputs, x)

    def basic_encode_layer(self, x, filters, activation='relu'):
        x = Conv2D(filters=filters, kernel_size=(5, 5),
                   strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        if activation == 'leaky_relu':
            x = LeakyReLU(0.2)(x)
        else:
            x = Activation(activation)(x)

        return x

    def basic_decode_layer(self, x, filters, activation='leaky_relu'):
        x = Conv2D(filters=filters, kernel_size=(5, 5), padding='same')(x)
        x = BatchNormalization()(x)
        if activation == 'leaky_relu':
            x = LeakyReLU(0.2)(x)
        else:
            x = Activation(activation)(x)

        x = UpSampling2D(size=(2, 2))(x)
        return x

    def classifier_loss(self, c_real, c_fake):
        def lossfun(c_data, unused):
            loss_real = K.mean(keras.metrics.binary_crossentropy(c_data, c_real), axis=-1)
            loss_fake = K.mean(keras.metrics.binary_crossentropy(c_data, c_fake), axis=-1)
            return loss_real - loss_fake

        return lossfun

    def discriminator_loss(self, y_real, y_fake):
        def lossfun(y0, y1):
            y_pos = K.ones_like(y_real)
            y_neg = K.zeros_like(y_real)
            loss_real = K.mean(keras.metrics.binary_crossentropy(y_pos, y_real), axis=-1)
            loss_fake = K.mean(keras.metrics.binary_crossentropy(y_neg, y_fake), axis=-1)
            return loss_real + loss_fake

        return lossfun

    def generator_loss(self,
        x_inputs, x_rec_from_x,
        y_pred_from_data, y_pred_from_x, y_pred_from_z,
        c_pred_from_data, c_pred_from_x, c_pred_from_z
    ):
        def lossfun(y0, y1):
            loss_GD = K.mean(K.square(y_pred_from_data - y_pred_from_z))
            loss_GC = K.mean(K.square(c_pred_from_data - c_pred_from_z))
            loss_G = K.mean(K.square(x_inputs - x_rec_from_x)) + \
                     K.mean(K.square(y_pred_from_data - y_pred_from_x)) + \
                     K.mean(K.square(c_pred_from_data - c_pred_from_x))

            return loss_GD + loss_GC + loss_G

        return lossfun

    def autoencoder_loss(self, x_true, x_pred, z_avg, z_log_var):
        def lossfun(y0, y1):
            size = K.shape(x_true)[1:]
            scale = K.cast(K.prod(size), 'float32')
            entropy = K.mean(keras.metrics.binary_crossentropy(x_true, x_pred)) * scale
            kl_loss = K.mean(-0.5 * K.sum(1.0 + z_log_var - K.square(z_avg) - K.exp(z_log_var), axis=-1))
            return entropy + kl_loss

        return lossfun
