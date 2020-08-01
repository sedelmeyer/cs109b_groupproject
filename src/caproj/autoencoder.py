"""
caproj.autoencoder
~~~~~~~~~~~~~~~~~~

This module contains functions for visualizing data and model results

**Module variables:**

.. autosummary::

   random_seed


**Module functions:**

.. autosummary::

   build_dense_ae_architecture
   plot_history

"""
from numpy.random import seed
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt

random_seed = 109
"""Import and set seed for reproducible results"""

seed(random_seed)

tf.random.set_seed(random_seed)


def build_dense_ae_architecture(
    input_dim, encoding_dim, droprate, learning_rate, name
):
    """Builds and compiles a tensorflow.keras dense autoencoder network

    NOTE:

       This network architecture was designed for the specific purpose of
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

    encoded = Dense(encoding_dim * 256, activation="relu", use_bias=True)(
        input_layer
    )
    encoded = Dropout(rate=droprate)(encoded)
    encoded = Dense(encoding_dim * 64, activation="relu", use_bias=True)(
        encoded
    )
    encoded = Dropout(rate=droprate)(encoded)
    encoded = Dense(encoding_dim * 16, activation="relu", use_bias=True)(
        encoded
    )
    encoded = Dropout(rate=droprate)(encoded)
    encoded = Dense(encoding_dim * 4, activation="relu", use_bias=True)(encoded)
    encoded = Dropout(rate=droprate)(encoded)
    encoded = Dense(encoding_dim, activation="linear", use_bias=True)(encoded)

    encoder = Model(input_layer, encoded, name="{}_encoder".format(name))

    # define decoder model
    latent_input = Input(shape=(encoding_dim,))

    decoded = Dense(encoding_dim * 4, activation="relu", use_bias=True)(
        latent_input
    )
    decoded = Dropout(rate=droprate)(decoded)
    decoded = Dense(encoding_dim * 16, activation="relu", use_bias=True)(
        decoded
    )
    decoded = Dropout(rate=droprate)(decoded)
    decoded = Dense(encoding_dim * 64, activation="relu", use_bias=True)(
        decoded
    )
    decoded = Dropout(rate=droprate)(decoded)
    decoded = Dense(encoding_dim * 256, activation="relu", use_bias=True)(
        decoded
    )
    decoded = Dropout(rate=droprate)(decoded)
    decoded = Dense(input_dim, activation="linear", use_bias=True)(decoded)

    decoder = Model(latent_input, decoded, name="{}_decoder".format(name))

    # define full non-linear autoencoder model
    ae = Sequential([encoder, decoder], name=name)

    # set loss, optimizer, and compile model
    loss = tf.keras.losses.mean_squared_error
    optimizer = Adam(lr=learning_rate)

    ae.compile(loss=loss, optimizer=optimizer)

    return ae, encoder, decoder


def plot_history(history, title, val_name="validation", loss_type="MSE"):
    """Plot training and validation loss using keras history object

    :param history: keras training history object or dict. If a dict is
                    used, it must have two keys named 'loss' and 'val_loss'
                    for which the corresponding values must be lists or
                    arrays with float values
    :param title: string, the title of the resulting plot
    :param val_name: string, the name for the val_loss line in the plot
                     legend (default 'validation')
    :param loss_type: string, the loss type name to be printed as the
                      y axis label (default 'MSE')
    :return: a line plot illustrating model training history, no
             objects are returned
    """
    if type(history) == dict:
        n_epochs = len(history["loss"])
        loss = history["loss"]
        val_loss = history["val_loss"]
    else:
        n_epochs = len(history.history["loss"])
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]

    x_vals = np.arange(1, n_epochs + 1)

    # adjust interval of x_ticks based on n_epochs
    if n_epochs < 40:
        x_ticks = x_vals
    elif n_epochs < 140:
        x_ticks = np.arange(0, n_epochs + 1, 5)
    else:
        x_ticks = np.arange(0, n_epochs + 1, 10)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    plt.suptitle("{}".format(title), fontsize=18, y=1)

    ax.plot(x_vals, loss, "k--", label="training")
    ax.plot(x_vals, val_loss, "k-", label=val_name)
    ax.set_xlabel("epoch", fontsize=14)
    ax.set_ylabel("loss ({})".format(loss_type), fontsize=14)
    ax.set_xticks(x_ticks)
    ax.grid(":", alpha=0.4)
    ax.tick_params(labelsize=12)

    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()
