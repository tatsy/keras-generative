import keras
from keras.layers import Conv2D, UpSampling2D, BatchNormalization, Conv2DTranspose
from keras.layers import Activation, ELU, LeakyReLU, Dropout

def BasicConvLayer(
    filters,
    kernel_size=(5, 5),
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

        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   kernel_initializer=kernel_init,
                   bias_initializer=bias_init,
                   padding='same')(x)

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

def BasicDeconvLayer(
    filters,
    kernel_size=(5, 5),
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
                            padding='same')(x)

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
