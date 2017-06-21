import keras
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Reshape
from keras.layers import Activation, LeakyReLU, ELU
from keras.layers import Conv2D, UpSampling2D, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K

def draw_normal(args):
    z_avg, z_log_var = args
    batch_size = K.shape(z_avg)[0]
    z_dims = K.shape(z_avg)[1]
    eps = K.random_normal(shape=(batch_size, z_dims), mean=0.0, stddev=1.0)
    return z_avg + K.exp(z_log_var / 2.0) * eps

class VAE(object):
    def __init__(self,
        input_shape=(64, 64, 3),
        z_dims = 128,
        enc_activation='sigmoid',
        dec_activation='sigmoid'
    ):
        self.input_shape = input_shape
        self.z_dims = z_dims
        self.enc_activation = enc_activation
        self.dec_activation = dec_activation

        self.encoder = None
        self.decoder = None

        self.build_model()

    def train_on_batch(self, x_batch):
        return self.trainer.train_on_batch(x_batch, x_batch)

    def predict(self, z_samples):
        return self.decoder.predict(z_samples)

    def build_model(self):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        inputs = Input(shape=self.input_shape)
        z_avg, z_log_var = self.encoder(inputs)
        z = Lambda(draw_normal, output_shape=(self.z_dims,))([z_avg, z_log_var])
        y = self.decoder(z)

        self.trainer = Model(inputs, y)
        self.trainer.compile(loss=self.variational_loss(z_avg, z_log_var),
                             optimizer=Adam(lr=2.0e-4, beta_1=0.5))

        self.trainer.summary()

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = self.basic_encoder_layer(inputs, filters=64)
        x = self.basic_encoder_layer(x, filters=128)
        x = self.basic_encoder_layer(x, filters=256)
        x = self.basic_encoder_layer(x, filters=256)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation('relu')(x)

        z_avg = Dense(self.z_dims)(x)
        z_log_var = Dense(self.z_dims)(x)
        x = Activation(self.enc_activation)(x)

        return Model(inputs, [z_avg, z_log_var], name='encoder')

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

    def variational_loss(self, z_avg, z_log_var):
        def lossfun(x_true, x_pred):
            size = K.shape(x_true)[1:]
            scale = K.cast(K.prod(size), 'float32')
            entropy = K.mean(keras.metrics.binary_crossentropy(x_true, x_pred)) * scale
            kl_loss = K.mean(-0.5 * K.sum(1.0 + z_log_var - K.square(z_avg) - K.exp(z_log_var), axis=-1))
            return entropy + kl_loss

        return lossfun
