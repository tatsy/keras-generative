import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import keras
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, Concatenate
from keras.layers import Activation, LeakyReLU, ELU
from keras.layers import Conv2D, UpSampling2D, BatchNormalization, Dropout
from keras.optimizers import Adam, Adadelta
from keras import backend as K

from .base import BaseModel, set_trainable
from .layers import *

def zero_loss(y_true, y_pred):
    return K.zeros_like(y_true)

class DiscriminatorLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(DiscriminatorLossLayer, self).__init__(**kwargs)

        # Parameters for BEGAN technique
        self.k_t = K.variable(0.0)
        self.lambda_k = K.variable(1.0e-3)
        self.new_k_t = K.variable(0.0)
        self.gamma = K.variable(0.5)

    def lossfun(self, y_real, y_real_rec, y_fake, y_fake_rec):
        loss_real = K.mean(K.abs(y_real - y_real_rec))
        loss_fake = K.mean(K.abs(y_fake - y_fake_rec))

        # Balancing learning speed with BEGAN technique
        self.new_k_t = K.clip(self.k_t + self.lambda_k * (self.gamma * loss_real - loss_fake), 0.0, 1.0)
        return loss_real - self.k_t * loss_fake

    def call(self, inputs):
        y_real = inputs[0]
        y_real_rec = inputs[1]
        y_fake = inputs[2]
        y_fake_rec = inputs[3]
        loss = self.lossfun(y_real, y_real_rec, y_fake, y_fake_rec)
        self.add_loss(loss, inputs=inputs)

        updates = []
        updates.append((self.k_t, self.new_k_t))
        self.add_update(updates, inputs)

        return y_real

class GeneratorLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(GeneratorLossLayer, self).__init__(**kwargs)

    def lossfun(self, x_fake, x_fake_rec, y_fake, y_fake_rec,
                x_input, x_cycle, y_input, y_cycle):
        x_loss = K.mean(K.abs(x_fake - x_fake_rec), axis=[1, 2, 3])
        y_loss = K.mean(K.abs(y_fake - y_fake_rec), axis=[1, 2, 3])
        x_cycle_loss = K.mean(K.abs(x_input - x_cycle), axis=[1, 2, 3])
        y_cycle_loss = K.mean(K.abs(y_input - y_cycle), axis=[1, 2, 3])
        return K.mean(x_loss + y_loss + x_cycle_loss + y_cycle_loss)

    def call(self, inputs):
        x_fake = inputs[0]
        x_fake_rec = inputs[1]
        y_fake = inputs[2]
        y_fake_rec = inputs[3]
        x_input = inputs[4]
        x_cycle = inputs[5]
        y_input = inputs[6]
        y_cycle = inputs[7]
        loss = self.lossfun(x_fake, x_fake_rec, y_fake, y_fake_rec,
                            x_input, x_cycle, y_input, y_cycle)
        self.add_loss(loss, inputs=inputs)

        return x_fake

class CycleGAN(BaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        z_dims=128,
        filters=128,
        name='cycle_gan',
        **kwargs
    ):
        super(CycleGAN, self).__init__(name=name, **kwargs)

        self.input_shape = input_shape
        self.z_dims = z_dims
        self.filters = filters

        self.f_gen_x2y = None
        self.f_gen_y2x = None
        self.f_dis_x = None
        self.f_dis_y = None

        self.gen_trainer = None,
        self.dis_x_trainer = None
        self.dis_y_trainer = None

        self.build_model()

    def train_on_batch(self, xy_data):
        x_data, y_data = xy_data
        dummy = np.zeros_like(x_data)

        g_loss = self.gen_trainer.train_on_batch([x_data, y_data], dummy)
        d_x_loss = self.dis_x_trainer.train_on_batch([x_data, y_data], dummy)
        d_y_loss = self.dis_x_trainer.train_on_batch([x_data, y_data], dummy)

        loss = {
            'g_loss': g_loss,
            'd_loss': 0.5 * (d_x_loss + d_y_loss),
        }
        return loss

    def make_batch(self, datasets, indx):
        x = datasets.x_datasets[indx]
        y = datasets.y_datasets[indx]
        return (x, y)

    def predict(self, x):
        return self.predict_x2y(x)

    def predict_x2y(self, x):
        return self.f_gen_x2y.predict(x)

    def predict_y2x(self, y):
        return self.f_gen_y2x.predict(x)

    def build_model(self):
        self.f_gen_x2y = self.build_autoencoder()
        self.f_gen_y2x = self.build_autoencoder()
        self.f_dis_x = self.build_discriminator()
        self.f_dis_y = self.build_discriminator()

        x_input = Input(shape=self.input_shape)
        y_input = Input(shape=self.input_shape)

        y_pred_from_x = self.f_gen_x2y(x_input)
        x_pred_from_y = self.f_gen_y2x(y_input)

        x_cycle = self.f_gen_y2x(y_pred_from_x)
        y_cycle = self.f_gen_x2y(x_pred_from_y)

        x_real_rec = self.f_dis_x(x_input)
        x_fake_rec = self.f_dis_x(x_pred_from_y)

        y_real_rec = self.f_dis_y(y_input)
        y_fake_rec = self.f_dis_y(y_pred_from_x)

        dis_x_loss = DiscriminatorLossLayer()([x_input, x_real_rec, x_pred_from_y, x_fake_rec])
        dis_y_loss = DiscriminatorLossLayer()([y_input, y_real_rec, y_pred_from_x, y_fake_rec])

        gen_loss = GeneratorLossLayer()(
            [x_pred_from_y, x_fake_rec, y_pred_from_x, y_fake_rec,
             x_input, x_cycle, y_input, y_cycle]
        )

        # Build discriminators
        set_trainable(self.f_gen_x2y, False)
        set_trainable(self.f_gen_y2x, False)
        set_trainable(self.f_dis_x, True)
        set_trainable(self.f_dis_y, False)

        self.dis_x_trainer = Model(
            inputs=[x_input, y_input],
            outputs=[dis_x_loss],
            name='dis_x_trainer'
        )
        self.dis_x_trainer.compile(
            loss=[zero_loss],
            optimizer=Adam(lr=2.0e-4, beta_1=0.5)
        )
        self.dis_x_trainer.summary()

        set_trainable(self.f_gen_x2y, False)
        set_trainable(self.f_gen_y2x, False)
        set_trainable(self.f_dis_x, False)
        set_trainable(self.f_dis_y, True)

        self.dis_y_trainer = Model(
            inputs=[x_input, y_input],
            outputs=[dis_y_loss],
            name='dis_y_trainer'
        )
        self.dis_y_trainer.compile(
            loss=[zero_loss],
            optimizer=Adam(lr=2.0e-4, beta_1=0.5)
        )
        self.dis_y_trainer.summary()

        # Build generators
        set_trainable(self.f_gen_x2y, True)
        set_trainable(self.f_gen_y2x, True)
        set_trainable(self.f_dis_x, False)
        set_trainable(self.f_dis_y, False)

        self.gen_trainer = Model(
            inputs=[x_input, y_input],
            outputs=[gen_loss],
            name='gen_trainer'
        )
        self.gen_trainer.compile(
            loss=[zero_loss],
            optimizer=Adam(lr=2.0e-4, beta_1=0.5)
        )
        self.gen_trainer.summary()

        self.store_to_save('gen_trainer')
        self.store_to_save('dis_x_trainer')
        self.store_to_save('dis_y_trainer')

    def build_autoencoder(self):
        inputs = Input(shape=self.input_shape)
        f_enc = self.build_encoder()
        f_dec = self.build_decoder()

        z = f_enc(inputs)
        x_rec = f_dec(z)

        return Model(inputs, x_rec)

    def build_discriminator(self):
        return self.build_autoencoder()

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        n = self.filters
        x = BasicConvLayer(filters=n, kernel_size=(5, 5), strides=(2, 2))(inputs)
        x = BasicConvLayer(filters=2*n, kernel_size=(5, 5), strides=(2, 2))(x)
        x = BasicConvLayer(filters=3*n, kernel_size=(5, 5), strides=(2, 2))(x)
        x = BasicConvLayer(filters=4*n, kernel_size=(5, 5), strides=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(self.z_dims)(x)
        x = Activation('linear')(x)

        return Model(inputs, x)

    def build_decoder(self):
        inputs = Input(shape=(self.z_dims,))

        n = self.filters
        x = Dense(4 * 4 * (4*n))(inputs)
        x = Reshape((4, 4, 4*n))(x)

        x = BasicDeconvLayer(filters=4*n, kernel_size=(5, 5), strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=3*n, kernel_size=(5, 5), strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=2*n, kernel_size=(5, 5), strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=n, kernel_size=(5, 5), strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=3, kernel_size=(5, 5), activation='tanh')(x)

        return Model(inputs, x)

    def save_images(self, gen, samples, filename):
        imgs = gen.predict(samples) * 0.5 + 0.5
        imgs = np.clip(imgs, 0.0, 1.0)

        fig = plt.figure(figsize=(8, 8))
        grid = gridspec.GridSpec(10, 10, wspace=0.1, hspace=0.1)
        for i in range(50):
            ax = plt.Subplot(fig, grid[i * 2 + 0])
            ax.imshow(samples[i] * 0.5 + 0.5, interpolation='none', vmin=0.0, vmax=1.0)
            ax.axis('off')
            fig.add_subplot(ax)

            ax = plt.Subplot(fig, grid[i * 2 + 1])
            ax.imshow(imgs[i], interpolation='none', vmin=0.0, vmax=1.0)
            ax.axis('off')
            fig.add_subplot(ax)

        fig.savefig(filename, dpi=200)
        plt.close(fig)
