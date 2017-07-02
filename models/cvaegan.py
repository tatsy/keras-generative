import os
import numpy as np

import keras
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, Concatenate
from keras.layers import Activation, LeakyReLU, ELU
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K

from .utils import set_trainable
from .cond_base import CondBaseModel

def sample_normal(args):
    z_avg, z_log_var = args
    batch_size = K.shape(z_avg)[0]
    z_dims = K.shape(z_avg)[1]
    eps = K.random_normal(shape=(batch_size, z_dims), mean=0.0, stddev=1.0)
    return z_avg + K.exp(z_log_var / 2.0) * eps

def zero_loss(y_true, y_pred):
    return K.zeros_like(y_true)


class ClassifierLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(ClassifierLossLayer, self).__init__(**kwargs)

    def lossfun(self, c_true, c_real, c_fake):
        loss_real = keras.metrics.binary_crossentropy(c_true, c_real)
        loss_fake = keras.metrics.binary_crossentropy(c_true, c_fake)
        return K.mean(loss_real - loss_fake)

    def call(self, inputs):
        c_true = inputs[0]
        c_real = inputs[1]
        c_fake = inputs[2]
        loss = self.lossfun(c_true, c_real, c_fake)
        self.add_loss(loss, inputs=inputs)

        return c_true

class DiscriminatorLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(DiscriminatorLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_real, y_fake):
        y_pos = K.ones_like(y_real)
        y_neg = K.zeros_like(y_real)
        loss_real = keras.metrics.binary_crossentropy(y_pos, y_real)
        loss_fake = keras.metrics.binary_crossentropy(y_neg, y_fake)
        return K.mean(loss_real + loss_fake)

    def call(self, inputs):
        y_real = inputs[0]
        y_fake = inputs[1]
        loss = self.lossfun(y_real, y_fake)
        self.add_loss(loss, inputs=inputs)

        return y_real

class GeneratorLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(GeneratorLossLayer, self).__init__(**kwargs)

    def lossfun(self,
        x_inputs, x_rec_from_x,
        y_pred_from_data, y_pred_from_x, y_pred_from_z,
        c_pred_from_data, c_pred_from_x, c_pred_from_z):

        loss_GD = K.mean(K.square(y_pred_from_data - y_pred_from_z))
        loss_GC = K.mean(K.square(c_pred_from_data - c_pred_from_z))
        loss_G = K.mean(K.square(x_inputs - x_rec_from_x)) + \
                 K.mean(K.square(y_pred_from_data - y_pred_from_x)) + \
                 K.mean(K.square(c_pred_from_data - c_pred_from_x))

        return loss_GD + loss_GC + loss_G

        return lossfun

    def call(self, inputs):
        x_inputs = inputs[0]
        x_rec_from_x = inputs[1]
        y_pred_from_data = inputs[2]
        y_pred_from_x = inputs[3]
        y_pred_from_z = inputs[4]
        c_pred_from_data = inputs[5]
        c_pred_from_x = inputs[6]
        c_pred_from_z = inputs[7]

        loss = self.lossfun(x_inputs, x_rec_from_x,
                            y_pred_from_data, y_pred_from_x, y_pred_from_z,
                            c_pred_from_data, c_pred_from_x, c_pred_from_z)
        self.add_loss(loss, inputs=inputs)

        return x_inputs

class AutoEncoderLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(AutoEncoderLossLayer, self).__init__(**kwargs)

    def lossfun(self, x_true, x_pred, z_avg, z_log_var):
        scale = K.cast(K.prod(K.shape(x_true)[1:]), 'float32')
        entropy = K.square(x_true - x_pred) * scale
        kl_loss = -0.5 * K.sum(1.0 + z_log_var - K.square(z_avg) - K.exp(z_log_var), axis=-1)
        return K.mean(entropy) + K.mean(kl_loss)

    def call(self, inputs):
        x_true = inputs[0]
        x_pred = inputs[1]
        z_avg = inputs[2]
        z_log_var = inputs[3]
        loss = self.lossfun(x_true, x_pred, z_avg, z_log_var)
        self.add_loss(loss, inputs=inputs)

        return x_true

def discriminator_accuracy(y_real, y_fake):
    def accfun(y0, y1):
        y_pos = K.ones_like(y_real)
        y_neg = K.ones_like(y_fake)
        loss_real = K.mean(keras.metrics.binary_accuracy(y_pos, y_real))
        loss_fake = K.mean(keras.metrics.binary_accuracy(y_neg, y_fake))
        return 0.5 * (loss_real + loss_fake)

    return accfun

def generator_accuracy(y_pred_from_x, y_pred_from_z):
    def accfun(y0, y1):
        y_pos = K.ones_like(y_pred_from_x)
        loss_x = K.mean(keras.metrics.binary_accuracy(y_pos, y_pred_from_x))
        loss_z = K.mean(keras.metrics.binary_accuracy(y_pos, y_pred_from_z))
        return 0.5 * (loss_x + loss_z)

    return accfun

class CVAEGAN(CondBaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        num_attrs=40,
        z_dims = 128,
        enc_activation='linear',
        dec_activation='tanh',
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
        g_loss, g_acc = self.gen_trainer.train_on_batch([x_image, z_rand, x_attr], x_image)

        # Generate fake sample
        x_fake = self.f_dec.predict_on_batch([z_rand, x_attr])

        # Train classifier
        c_loss = self.cls_trainer.train_on_batch([x_image, x_fake, x_attr], x_attr)

        # Train discriminator
        y_pos = np.ones((batchsize, self.z_dims), dtype='float32')
        d_loss, d_acc = self.dis_trainer.train_on_batch([x_image, x_fake], y_pos)

        loss = {
            'g_loss': g_loss,
            'd_loss': d_loss,
            'g_acc': g_acc,
            'd_acc': d_acc
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
        c_true = Input(shape=(self.num_attrs,))
        c_pred_real = self.f_cls(x_real)
        c_pred_fake = self.f_cls(x_fake)

        c_loss = ClassifierLossLayer()([c_true, c_pred_real, c_pred_fake])

        self.cls_trainer = Model(inputs=[x_real, x_fake, c_true],
                                 outputs=c_loss)
        self.cls_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=1.0e-5, beta_1=0.5))
        self.cls_trainer.summary()

        # Build discriminator trainer
        y_pred_real = self.f_dis(x_real)
        y_pred_fake = self.f_dis(x_fake)

        d_loss = DiscriminatorLossLayer()([y_pred_real, y_pred_fake])

        self.dis_trainer = Model(inputs=[x_real, x_fake],
                                 outputs=d_loss)
        self.dis_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=1.0e-5, beta_1=0.5),
                                 metrics=[discriminator_accuracy(y_pred_real, y_pred_fake)])
        self.dis_trainer.summary()

        # Build generator trainer
        set_trainable(self.f_cls, False)
        set_trainable(self.f_dis, False)
        set_trainable(self.f_enc, False)

        x_inputs = Input(shape=self.input_shape)
        c_inputs = Input(shape=(self.num_attrs,))
        z_inputs = Input(shape=(self.z_dims,))

        z_params = self.f_enc([x_inputs, c_inputs])
        z_avg = Lambda(lambda x: x[:, :self.z_dims], output_shape=(self.z_dims,))(z_params)
        z_log_var = Lambda(lambda x: x[:, self.z_dims:], output_shape=(self.z_dims,))(z_params)

        z_from_x = Lambda(sample_normal, output_shape=(self.z_dims,))([z_avg, z_log_var])
        x_rec_from_x = self.f_dec([z_from_x, c_inputs])
        x_rec_from_z = self.f_dec([z_inputs, c_inputs])

        y_pred_from_x = self.f_dis(x_rec_from_x)
        y_pred_from_z = self.f_dis(x_rec_from_z)
        y_pred_from_data = self.f_dis(x_inputs)

        c_pred_from_x = self.f_cls(x_rec_from_x)
        c_pred_from_z = self.f_cls(x_rec_from_z)
        c_pred_from_data = self.f_cls(x_inputs)

        g_loss = GeneratorLossLayer()([x_inputs, x_rec_from_x,
                                       y_pred_from_data, y_pred_from_x, y_pred_from_z,
                                       c_pred_from_data, c_pred_from_x, c_pred_from_z])

        self.gen_trainer = Model(inputs=[x_inputs, z_inputs, c_inputs],
                                 outputs=[g_loss])
        self.gen_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=1.0e-4, beta_1=0.5),
                                 metrics=[generator_accuracy(y_pred_from_x, y_pred_from_z)])

        # Build autoencoder
        set_trainable(self.f_dec, False)
        set_trainable(self.f_enc, True)

        ae_loss = AutoEncoderLossLayer()([x_inputs, x_rec_from_x, z_avg, z_log_var])

        self.ae_trainer = Model(inputs=[x_inputs, c_inputs],
                                outputs=[ae_loss])
        self.ae_trainer.compile(loss=[zero_loss],
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

        x = self.basic_encode_layer(inputs, filters=128)
        x = self.basic_encode_layer(x, filters=256)
        x = self.basic_encode_layer(x, filters=256)
        x = self.basic_encode_layer(x, filters=512)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation('relu')(x)

        x = Dense(1)(x)
        x = Activation(self.dis_activation)(x)

        return Model(inputs, x)

    def build_classifier(self):
        inputs = Input(shape=self.input_shape)

        x = self.basic_encode_layer(inputs, filters=128)
        x = self.basic_encode_layer(x, filters=256)
        x = self.basic_encode_layer(x, filters=512)
        x = self.basic_encode_layer(x, filters=512)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation('relu')(x)

        x = Dense(self.num_attrs)(x)
        x = Activation(self.dis_activation)(x)

        return Model(inputs, x)

    def basic_encode_layer(self, x, filters, activation='leaky_relu'):
        x = Conv2D(filters=filters, kernel_size=(5, 5),
                   strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        if activation == 'leaky_relu':
            x = LeakyReLU(0.1)(x)
        else:
            x = Activation(activation)(x)

        return x

    def basic_decode_layer(self, x, filters, activation='leaky_relu'):
        x = Conv2DTranspose(filters=filters, kernel_size=(5, 5),
                            strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        if activation == 'leaky_relu':
            x = LeakyReLU(0.1)(x)
        else:
            x = Activation(activation)(x)

        return x
