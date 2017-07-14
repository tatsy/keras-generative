import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import keras
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers.merge import Add
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, Concatenate
from keras.layers import Activation, LeakyReLU, ELU
from keras.layers import Conv2D, UpSampling2D, BatchNormalization, Dropout
from keras.optimizers import Adam, Adadelta
from keras import backend as K

from .base import BaseModel, set_trainable
from .layers import *

def zero_loss(y_true, y_pred):
    return K.zeros_like(y_true)

def sample_normals(args):
    z_avg, z_log_var = args
    z_shape = K.shape(z_avg)

    eps = K.random_normal(shape=z_shape, mean=0.0, stddev=1.0)
    return z_avg + K.exp(z_log_var * 0.5) * eps

class DiscriminatorLossLayer(Layer):
    def __init__(self, batchsize, **kwargs):
        self.is_placeholder = True
        self.batchsize = batchsize
        super(DiscriminatorLossLayer, self).__init__(**kwargs)

        # Parameters for BEGAN technique
        self.k_t = K.variable(0.0)
        self.lambda_k = K.variable(1.0e-3)
        self.new_k_t = K.variable(0.0)
        self.gamma = K.variable(0.5)

    def lossfun(self, y_real, y_fake1, y_fake2):
        y_pos = K.ones_like(y_real)
        y_neg = K.zeros_like(y_real)

        loss_real = K.mean(keras.metrics.binary_crossentropy(y_pos, y_real))
        loss_fake1 = K.mean(keras.metrics.binary_crossentropy(y_neg, y_fake1))
        loss_fake2 = K.mean(keras.metrics.binary_crossentropy(y_neg, y_fake2))

        return loss_real + loss_fake1 + loss_fake2

    def call(self, inputs):
        y_real = inputs[0]
        y_fake1 = inputs[1]
        y_fake2 = inputs[2]
        loss = self.lossfun(y_real, y_fake1, y_fake2)
        self.add_loss(loss, inputs=inputs)

        # updates = []
        # updates.append((self.k_t, self.new_k_t))
        # self.add_update(updates, inputs)

        return y_real

class VariationalLossLayer(Layer):
    def __init__(self, batchsize, **kwargs):
        self.is_placeholder = True
        self.batchsize = batchsize
        super(VariationalLossLayer, self).__init__(**kwargs)

    def lossfun(self, z_avg, z_log_var):
        kl_loss = K.mean(-0.5 * K.sum(1.0 + z_log_var - K.square(z_avg) - K.exp(z_log_var)))
        return kl_loss

    def call(self, inputs):
        z1_avg = inputs[0]
        z1_log_var = inputs[1]
        z2_avg = inputs[2]
        z2_log_var = inputs[3]
        z3_avg = inputs[4]
        z3_log_var = inputs[5]
        z4_avg = inputs[6]
        z4_log_var = inputs[7]

        l1 = self.lossfun(z1_avg, z1_log_var)
        l2 = self.lossfun(z2_avg, z2_log_var)
        l3 = self.lossfun(z3_avg, z3_log_var)
        l4 = self.lossfun(z4_avg, z4_log_var)
        loss = (l1 + l2 + l3 + l4)
        self.add_loss(loss, inputs=inputs)

        return z1_avg

class MSELossLayer(Layer):
    def __init__(self, batchsize, **kwargs):
        self.is_placeholder = True
        self.batchsize = batchsize
        super(MSELossLayer, self).__init__(**kwargs)

    def lossfun(self, x_org, x_rec):
        return K.mean(K.square(x_org - x_rec))

    def call(self, inputs):
        x_org = inputs[0]
        x_rec = inputs[1]
        loss = self.lossfun(x_org, x_rec)
        self.add_loss(loss, inputs=inputs)

        return x_org

class FeatureMatchLayer(Layer):
    def __init__(self, batchsize, **kwargs):
        self.is_placeholder = True
        self.batchsize = batchsize
        super(FeatureMatchLayer, self).__init__(**kwargs)

    def lossfun(self, x_z, y_z):
        return K.mean(K.square(x_z - y_z))

    def call(self, inputs):
        x_z1 = inputs[0]
        x_z2 = inputs[1]
        x_z3 = inputs[2]
        x_z4 = inputs[3]
        y_z1 = inputs[4]
        y_z2 = inputs[5]
        y_z3 = inputs[6]
        y_z4 = inputs[7]
        l1 = self.lossfun(x_z1, y_z1)
        l2 = self.lossfun(x_z2, y_z2)
        l3 = self.lossfun(x_z3, y_z3)
        l4 = self.lossfun(x_z4, y_z4)
        loss = 0.5 * (l1 + l2 + l3 + l4)
        self.add_loss(loss, inputs=inputs)

        return x_z1

class GeneratorLossLayer(Layer):
    def __init__(self, batchsize, **kwargs):
        self.is_placeholder = True
        self.batchsize = batchsize
        super(GeneratorLossLayer, self).__init__(**kwargs)

    def lossfun(self, x_fake1, x_fake2):
        y_pos = K.ones_like(x_fake1)
        loss_fake1 = K.mean(keras.metrics.binary_crossentropy(y_pos, x_fake1))
        loss_fake2 = K.mean(keras.metrics.binary_crossentropy(y_pos, x_fake2))
        return loss_fake1 + loss_fake2

    def call(self, inputs):
        x_fake1 = inputs[0]
        x_fake2 = inputs[1]
        loss = self.lossfun(x_fake1, x_fake2)
        self.add_loss(loss, inputs=inputs)

        return x_fake1

class UNIT(BaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        z_dims=128,
        filters=128,
        batchsize=50,
        name='unit',
        **kwargs
    ):
        super(UNIT, self).__init__(name=name, **kwargs)

        self.input_shape = input_shape
        self.z_dims = z_dims
        self.filters = filters
        self.batchsize = batchsize

        self.f_enc_x_in = None
        self.f_enc_y_in = None
        self.f_dec_x_out = None
        self.f_dec_y_out = None
        self.f_dis_x = None
        self.f_dis_y = None

        self.gen_x_trainer = None
        self.gen_y_trainer = None
        self.dis_x_trainer = None
        self.dis_y_trainer = None

        self.build_model()

    def train_on_batch(self, xy_data):
        x_data, y_data = xy_data
        batchsize = len(x_data)
        _, h, w, dims = x_data.shape
        dummy1 = np.zeros((batchsize, h, w, 3), dtype='float32')
        dummy2 = np.zeros((batchsize), dtype='float32')
        dummy3 = np.zeros((batchsize, h//4, w//4, 128), dtype='float32')

        g_x_loss, _, _, _, _ = self.gen_x_trainer.train_on_batch([x_data, y_data], [dummy3, dummy1, dummy2, dummy3])
        g_y_loss, _, _, _, _ = self.gen_y_trainer.train_on_batch([x_data, y_data], [dummy3, dummy1, dummy2, dummy3])
        d_x_loss = self.dis_x_trainer.train_on_batch([x_data, y_data], dummy2)
        d_y_loss = self.dis_y_trainer.train_on_batch([x_data, y_data], dummy2)

        loss = {
            'g_loss': 0.5 * (g_x_loss + g_y_loss),
            'd_loss': 0.5 * (d_x_loss + d_y_loss)
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
        self.f_enc_x_in, self.f_enc_y_in = self.build_pair_of_encoders()
        self.f_dec_x_out, self.f_dec_y_out = self.build_pair_of_decoders()
        self.f_dis_x, self.f_dis_y = self.build_pair_of_discriminators()

        x_input = Input(shape=self.input_shape)
        y_input = Input(shape=self.input_shape)

        vx_loss, x_z1, x_z2, x_z3, x_z4 = self.f_enc_x_in(x_input)
        vy_loss, y_z1, y_z2, y_z3, y_z4 = self.f_enc_y_in(y_input)

        fm_loss = FeatureMatchLayer(self.batchsize)(
            [x_z1, x_z2, x_z3, x_z4, y_z1, y_z2, y_z3, y_z4]
        )

        x_fake_from_x = self.f_dec_x_out([x_z1, x_z2, x_z3, x_z4])
        x_fake_from_y = self.f_dec_x_out([y_z1, y_z2, y_z3, y_z4])
        y_fake_from_x = self.f_dec_y_out([x_z1, x_z2, x_z3, x_z4])
        y_fake_from_y = self.f_dec_y_out([y_z1, y_z2, y_z3, y_z4])

        t_x_true = self.f_dis_x(x_input)
        t_x_ffx = self.f_dis_x(x_fake_from_x)
        t_x_ffy = self.f_dis_x(x_fake_from_y)
        t_y_true = self.f_dis_y(y_input)
        t_y_ffx = self.f_dis_y(y_fake_from_x)
        t_y_ffy = self.f_dis_y(y_fake_from_y)

        x_mse_loss = MSELossLayer(self.batchsize)([x_input, x_fake_from_x])
        y_mse_loss = MSELossLayer(self.batchsize)([y_input, y_fake_from_y])

        gx_loss = GeneratorLossLayer(self.batchsize)([t_x_ffx, t_x_ffy])
        gy_loss = GeneratorLossLayer(self.batchsize)([t_y_ffx, t_y_ffy])

        dx_loss = DiscriminatorLossLayer(self.batchsize)([t_x_true, t_x_ffx, t_x_ffy])
        dy_loss = DiscriminatorLossLayer(self.batchsize)([t_y_true, t_y_ffx, t_y_ffy])

        # Build discriminators
        set_trainable(self.f_enc_x_in, False)
        set_trainable(self.f_enc_y_in, False)
        set_trainable(self.f_dec_x_out, False)
        set_trainable(self.f_dec_y_out, False)
        set_trainable(self.f_dis_x, True)
        set_trainable(self.f_dis_y, True)

        self.dis_x_trainer = Model(
            inputs=[x_input, y_input],
            outputs=[dx_loss],
            name='dis_x_trainer'
        )
        self.dis_x_trainer.compile(
            loss=[zero_loss],
            optimizer=Adam(lr=2.0e-4, beta_1=0.5)
        )
        self.dis_x_trainer.summary()

        self.dis_y_trainer = Model(
            inputs=[x_input, y_input],
            outputs=[dy_loss],
            name='dis_y_trainer'
        )
        self.dis_y_trainer.compile(
            loss=[zero_loss],
            optimizer=Adam(lr=2.0e-4, beta_1=0.5)
        )
        self.dis_y_trainer.summary()

        # Build generators
        set_trainable(self.f_enc_x_in, True)
        set_trainable(self.f_enc_y_in, True)
        set_trainable(self.f_dec_x_out, True)
        set_trainable(self.f_dec_y_out, True)
        set_trainable(self.f_dis_x, False)
        set_trainable(self.f_dis_y, False)

        self.gen_x_trainer = Model(
            inputs=[x_input, y_input],
            outputs=[vx_loss, x_mse_loss, gx_loss, fm_loss],
            name='gen_x_trainer'
        )
        self.gen_x_trainer.compile(
            loss=[zero_loss, zero_loss, zero_loss, zero_loss],
            optimizer=Adam(lr=2.0e-4, beta_1=0.5)
        )
        self.gen_x_trainer.summary()

        self.gen_y_trainer = Model(
            inputs=[x_input, y_input],
            outputs=[vy_loss, y_mse_loss, gy_loss, fm_loss],
            name='gen_y_trainer'
        )
        self.gen_y_trainer.compile(
            loss=[zero_loss, zero_loss, zero_loss, zero_loss],
            optimizer=Adam(lr=2.0e-4, beta_1=0.5)
        )
        self.gen_y_trainer.summary()

        self.store_to_save('gen_x_trainer')
        self.store_to_save('gen_y_trainer')
        self.store_to_save('dis_x_trainer')
        self.store_to_save('dis_y_trainer')

        # Build x <--> y generator
        self.f_gen_x2y = Model(x_input, y_fake_from_x)
        self.f_gen_y2x = Model(y_input, x_fake_from_y)

    def build_pair_of_discriminators(self):
        x_input = Input(shape=self.input_shape)
        y_input = Input(shape=self.input_shape)

        shared_layers = self.build_shared_discriminator()

        x = BasicConvLayer(filters=64, kernel_size=(5, 5), strides=(2, 2))(x_input)
        x = BasicConvLayer(filters=128, kernel_size=(5, 5), strides=(2, 2))(x)
        x = shared_layers(x)

        dis_x = Model(x_input, x)

        y = BasicConvLayer(filters=64, kernel_size=(5, 5), strides=(2, 2))(y_input)
        y = BasicConvLayer(filters=128, kernel_size=(5, 5), strides=(2, 2))(y)
        y = shared_layers(y)

        dis_y = Model(y_input, y)

        return dis_x, dis_y

    def build_shared_discriminator(self):
        h, w, dims = self.input_shape
        inputs = Input(shape=(h//4, w//4, 128))

        x = BasicConvLayer(filters=256, kernel_size=(5, 5), strides=(2, 2))(inputs)
        x = BasicConvLayer(filters=512, kernel_size=(5, 5), strides=(2, 2))(x)
        x = BasicConvLayer(filters=1024, kernel_size=(5, 5), strides=(2, 2))(x)
        x = BasicConvLayer(filters=2048, kernel_size=(5, 5), strides=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        return Model(inputs, x)

    def build_pair_of_encoders(self):
        x_input = Input(shape=self.input_shape)
        y_input = Input(shape=self.input_shape)

        shared_layers = self.build_shared_encoder()

        x = BasicConvLayer(filters=64, kernel_size=(5, 5), strides=(2, 2))(x_input)
        x = BasicConvLayer(filters=128, kernel_size=(5, 5), strides=(2, 2))(x)
        v_losses, z1, z2, z3, z4 = shared_layers(x)

        enc_x = Model(inputs=x_input,
                      outputs=[v_losses, z1, z2, z3, z4])

        y = BasicConvLayer(filters=64, kernel_size=(5, 5), strides=(2, 2))(y_input)
        y = BasicConvLayer(filters=128, kernel_size=(5, 5), strides=(2, 2))(y)
        v_losses, z1, z2, z3, z4 = shared_layers(y)

        enc_y = Model(inputs=y_input,
                      outputs=[v_losses, z1, z2, z3, z4])

        return enc_x, enc_y

    def build_shared_encoder(self):
        h, w, dims = self.input_shape
        inputs = Input(shape=(h//4, w//4, 128))

        h1_avg = BasicConvLayer(filters=256, kernel_size=(1, 1), strides=(1, 1), bnorm=False, activation='linear')(inputs)
        h1_log_var = BasicConvLayer(filters=256, kernel_size=(1, 1), strides=(1, 1), bnorm=False, activation='softplus')(inputs)

        x = BasicConvLayer(filters=256, kernel_size=(5, 5), strides=(2, 2))(inputs)

        h2_avg = BasicConvLayer(filters=512, kernel_size=(1, 1), strides=(1, 1), bnorm=False, activation='linear')(x)
        h2_log_var = BasicConvLayer(filters=512, kernel_size=(1, 1), strides=(1, 1), bnorm=False, activation='softplus')(x)

        x = BasicConvLayer(filters=512, kernel_size=(5, 5), strides=(2, 2))(x)

        h3_avg = BasicConvLayer(filters=1024, kernel_size=(1, 1), strides=(1, 1), bnorm=False, activation='linear')(x)
        h3_log_var = BasicConvLayer(filters=1024, kernel_size=(1, 1), strides=(1, 1), bnorm=False, activation='softplus')(x)

        x = BasicConvLayer(filters=1024, kernel_size=(5, 5), strides=(2, 2))(x)

        h4_avg = BasicConvLayer(filters=2048, kernel_size=(1, 1), strides=(1, 1), bnorm=False, activation='linear')(x)
        h4_log_var = BasicConvLayer(filters=2048, kernel_size=(1, 1), strides=(1, 1), bnorm=False, activation='softplus')(x)

        z4 = Lambda(lambda z: sample_normals(z))([h4_avg, h4_log_var])
        z3 = Lambda(lambda z: sample_normals(z))([h3_avg, h3_log_var])
        z2 = Lambda(lambda z: sample_normals(z))([h2_avg, h2_log_var])
        z1 = Lambda(lambda z: sample_normals(z))([h1_avg, h1_log_var])

        v_losses = VariationalLossLayer(self.batchsize)(
            [h1_avg, h1_log_var, h2_avg, h2_log_var,
             h3_avg, h3_log_var, h4_avg, h4_log_var]
        )

        return Model(inputs=[inputs],
                     outputs=[v_losses, z1, z2, z3, z4])

    def build_pair_of_decoders(self):
        h, w, dims = self.input_shape
        z1_input = Input(shape=(h//4, w//4, 256))
        z2_input = Input(shape=(h//8, w//8, 512))
        z3_input = Input(shape=(h//16, w//16, 1024))
        z4_input = Input(shape=(h//32, w//32, 2048))

        shared_layers = self.build_shared_decoder()

        x = shared_layers([z1_input, z2_input, z3_input, z4_input])
        x = BasicDeconvLayer(filters=64, kernel_size=(5, 5), strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=3, kernel_size=(1, 1), strides=(1, 1), bnorm=False, activation='tanh')(x)

        dec_x = Model(inputs=[z1_input, z2_input, z3_input, z4_input],
                      outputs=[x])

        y = shared_layers([z1_input, z2_input, z3_input, z4_input])
        y = BasicDeconvLayer(filters=64, kernel_size=(5, 5), strides=(2, 2))(y)
        y = BasicDeconvLayer(filters=3, kernel_size=(1, 1), strides=(1, 1), bnorm=False, activation='tanh')(y)

        dec_y = Model(inputs=[z1_input, z2_input, z3_input, z4_input],
                      outputs=[y])

        return dec_x, dec_y

    def build_shared_decoder(self):
        h, w, dims = self.input_shape
        z1_input = Input(shape=(h//4, w//4, 256))
        z2_input = Input(shape=(h//8, w//8, 512))
        z3_input = Input(shape=(h//16, w//16, 1024))
        z4_input = Input(shape=(h//32, w//32, 2048))

        z4 = BasicDeconvLayer(filters=1024, kernel_size=(5, 5), strides=(2, 2))(z4_input)

        z3 = Add()([z3_input, z4])
        z3 = BasicDeconvLayer(filters=512, kernel_size=(5, 5), strides=(2, 2))(z3)

        z2 = Add()([z2_input, z3])
        z2 = BasicDeconvLayer(filters=256, kernel_size=(5, 5), strides=(2, 2))(z2)

        z1 = Add()([z1_input, z2])
        z1 = BasicDeconvLayer(filters=128, kernel_size=(5, 5), strides=(2, 2))(z1)

        return Model(inputs=[z1_input, z2_input, z3_input, z4_input],
                     outputs=[z1])

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
