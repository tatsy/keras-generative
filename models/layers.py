import keras
from keras.engine.topology import Layer
from keras.layers import Conv2D, UpSampling2D, BatchNormalization, Conv2DTranspose
from keras.layers import Activation, ELU, LeakyReLU, Dropout, Lambda
from keras import backend as K

class SampleNormal(Layer):
    __name__ = 'sample_normal'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(SampleNormal, self).__init__(**kwargs)

    def _sample_normal(self, z_avg, z_log_var):
        batch_size = K.shape(z_avg)[0]
        z_dims = K.shape(z_avg)[1]
        eps = K.random_normal(shape=K.shape(z_avg), mean=0.0, stddev=1.0)
        return z_avg + K.exp(z_log_var / 2.0) * eps

    def call(self, inputs):
        z_avg = inputs[0]
        z_log_var = inputs[1]
        return self._sample_normal(z_avg, z_log_var)

class MinibatchDiscrimination(Layer):
    __name__ = 'minibatch_discrimination'

    def __init__(self, kernels=50, dims=5, **kwargs):
        super(MinibatchDiscrimination, self).__init__(**kwargs)
        self.kernels = kernels
        self.dims = dims

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.W = self.add_weight(name='kernel',
                                 shape=(input_shape[1], self.kernels * self.dims),
                                 initializer='uniform',
                                 trainable=True)

    def call(self, inputs):
        Ms = K.dot(inputs, self.W)
        Ms = K.reshape(Ms, (-1, self.kernels, self.dims))
        x_i = K.reshape(Ms, (-1, self.kernels, 1, self.dims))
        x_j = K.reshape(Ms, (-1, 1, self.kernels, self.dims))
        x_i = K.repeat_elements(x_i, self.kernels, 2)
        x_j = K.repeat_elements(x_j, self.kernels, 1)
        norm = K.sum(K.abs(x_i - x_j), axis=3)
        Os = K.sum(K.exp(-norm), axis=2)
        return Os

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.kernels)

def BasicConvLayer(
    filters,
    kernel_size=(5, 5),
    padding='same',
    strides=(1, 1),
    bnorm=True,
    dropout=0.0,
    activation='leaky_relu'):

    def fun(inputs):
        kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        bias_init = keras.initializers.Zeros()

        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   kernel_initializer=kernel_init,
                   bias_initializer=bias_init,
                   padding=padding)(inputs)

        if bnorm:
            x = BatchNormalization()(x)

        if activation == 'leaky_relu':
            x = LeakyReLU(0.1)(x)
        elif activation == 'elu':
            x = ELU()(x)
        else:
            x = Activation(activation)(x)

        if dropout > 0.0:
            x = Dropout(dropout)(x)

        return x

    return fun

def BasicDeconvLayer(
    filters,
    kernel_size=(5, 5),
    padding='same',
    strides=(1, 1),
    bnorm=True,
    dropout=0.0,
    activation='leaky_relu'):

    def fun(inputs):
        if dropout > 0.0:
            x = Dropout(dropout)(inputs)
        else:
            x = inputs

        kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        bias_init = keras.initializers.Zeros()

        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            kernel_initializer=kernel_init,
                            bias_initializer=bias_init,
                            padding=padding)(x)

        if bnorm:
            x = BatchNormalization()(x)

        if activation == 'leaky_relu':
            x = LeakyReLU(0.1)(x)
        elif activation == 'elu':
            x = ELU()(x)
        else:
            x = Activation(activation)(x)

        return x

    return fun

class VAELossLayer(Layer):
    __name__ = 'vae_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(VAELossLayer, self).__init__(**kwargs)

    def lossfun(self, x_true, x_pred, z_avg, z_log_var):
        rec_loss = K.mean(K.square(x_true - x_pred))
        kl_loss = K.mean(-0.5 * K.sum(1.0 + z_log_var - K.square(z_avg) - K.exp(z_log_var), axis=-1))
        return rec_loss + kl_loss

    def call(self, inputs):
        x_true = inputs[0]
        x_pred = inputs[1]
        z_avg = inputs[2]
        z_log_var = inputs[3]
        loss = self.lossfun(x_true, x_pred, z_avg, z_log_var)
        self.add_loss(loss, inputs=inputs)

        return x_true
