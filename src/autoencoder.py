"""
This module contains functions for visualizing data and model results

PARAMETERS

    random_seed = 109
        This module sets a random seed for numpy.random.seed() and
        tensorflow.random.set_seed() to help ensure reproducible results

FUNCTIONS

    build_dense_ae_architecture()
        Builds and compiles a tensorflow.keras dense autoencoder network
"""

# import and set seeds for reproducible results
random_seed = 109

from numpy.random import seed
seed(random_seed)

import tensorflow as tf
tf.random.set_seed(random_seed)

# import remaining imports
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_dense_ae_architecture(input_dim, encoding_dim, droprate,
                                learning_rate, name):
    """Builds and compiles a tensorflow.keras dense autoencoder network

    NOTE: This network architecture was designed for the specific purpose of
          encoding a dataset of 1D embeddings. Therefore, the input dimension
          must be 1D with a length that equals the number of values in any
          single observation's embedding

    :param input_dim: integer, the length of each embedding (must all be of
                      the same length)
    :param encoding_dim: integer, the desired bottleneck dimension for the
                         encoder network
    :param droprate: float >0 <1, this is passed to the rate argument for
                     the dropout layers between each dense layer
    :param learning_rate: float, the desired learning rate for the Adam
                          optimizer used while compiling the model
    :param name: string, the desired name of the resulting network

    :return: tuple of 3 tf.keras model object, [0] full autoencoder model,
             [1] encoder model, [2] decoder model
    """
    # define encoder model
    input_layer = Input(shape=input_dim)
    
    encoded = Dense(encoding_dim*256, activation='relu', use_bias=True)(input_layer)
    encoded = Dropout(rate=droprate)(encoded)
    encoded = Dense(encoding_dim*64, activation='relu', use_bias=True)(encoded)
    encoded = Dropout(rate=droprate)(encoded)
    encoded = Dense(encoding_dim*16, activation='relu', use_bias=True)(encoded)
    encoded = Dropout(rate=droprate)(encoded)
    encoded = Dense(encoding_dim*4, activation='relu', use_bias=True)(encoded)
    encoded = Dropout(rate=droprate)(encoded)
    encoded = Dense(encoding_dim, activation='linear', use_bias=True)(encoded)
    
    encoder = Model(input_layer, encoded, name='{}_encoder'.format(name))

    # define decoder model
    latent_input = Input(shape=(encoding_dim,))
    
    decoded = Dense(encoding_dim*4, activation='relu', use_bias=True)(latent_input)
    decoded = Dropout(rate=droprate)(decoded)
    decoded = Dense(encoding_dim*16, activation='relu', use_bias=True)(decoded)
    decoded = Dropout(rate=droprate)(decoded)
    decoded = Dense(encoding_dim*64, activation='relu', use_bias=True)(decoded)
    decoded = Dropout(rate=droprate)(decoded)
    decoded = Dense(encoding_dim*256, activation='relu', use_bias=True)(decoded)
    decoded = Dropout(rate=droprate)(decoded)
    decoded = Dense(input_dim, activation='linear', use_bias=True)(decoded)
    
    decoder = Model(latent_input, decoded, name='{}_decoder'.format(name))

    # define full non-linear autoencoder model
    ae = Sequential(
        [
            encoder,
            decoder,
        ], name=name
    )

    # set loss, optimizer, and compile model
    loss = tf.keras.losses.mean_squared_error
    optimizer = Adam(lr=learning_rate)

    ae.compile(
        loss=loss,
        optimizer=optimizer
    )

    return ae, encoder, decoder
