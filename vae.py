import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Layer, Activation, GaussianDropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import metrics, optimizers
from keras.callbacks import Callback
import keras
import numpy as np
# import pydot
# import graphviz
# from keras.utils import plot_model
# from keras_tqdm import tqdmnotebookcallback
# from ipython.display import svg
# from keras.utils.vis_utils import model_to_dot

print(keras.__version__)
tf.__version__

# Define metrics functions
def mean_absolute_error_array(y_true, y_pred):
	return np.mean(np.abs((y_pred - y_true)))/np.mean(y_true)

def std_absolute_error_array(y_true, y_pred):
	return np.std(np.abs((y_pred - y_true)))/np.std(y_true)

class WarmUpCallback(Callback):
    def __init__(self, beta, kappa):
        self.beta = beta
        self.kappa = kappa

    # Behavior on each epoch
    def on_epoch_end(self, epoch, logs={}):
        if K.get_value(self.beta) <= 1:
            K.set_value(self.beta, K.get_value(self.beta) + self.kappa)


def get_vae(original_dim,
            latent_dim,
            intermediate_dim,
            epsilon_std=1.0,
            learning_rate=0.0005,
            beta=K.variable(0)):
    # Function for reparameterization trick to make model differentiable
    def sampling(args):
        import tensorflow as tf
        # Function with args required for Keras Lambda function
        z_mean, z_log_var = args

        # Draw epsilon of the same shape from a standard normal distribution
        epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0.,
                                  stddev=epsilon_std)

        # The latent vector is non-deterministic and differentiable
        # in respect to z_mean and z_log_var
        z = z_mean + K.exp(z_log_var / 2) * epsilon
        return z

    class CustomVariationalLayer(Layer):
        """
        Define a custom layer that learns and performs the training
        This function is borrowed from:
        https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py
        """

        def __init__(self, **kwargs):
            # https://keras.io/layers/writing-your-own-keras-layers/
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)

        def vae_loss(self, x_input, x_decoded):
            reconstruction_loss = original_dim * metrics.binary_crossentropy(x_input, x_decoded)
            kl_loss = - 0.5 * K.sum(1 + z_log_var_encoded - K.square(z_mean_encoded) -
                                    K.exp(z_log_var_encoded), axis=-1)
            return K.mean(reconstruction_loss + (K.get_value(beta) * kl_loss))

        def call(self, inputs):
            x = inputs[0]
            x_decoded = inputs[1]
            loss = self.vae_loss(x, x_decoded)
            self.add_loss(loss, inputs=inputs)
            # We won't actually use the output.
            return x

    # Encoder
    rnaseq_input = Input(shape=(original_dim,))
    z_mean_dense_linear = Dense(latent_dim, kernel_initializer='glorot_uniform')(rnaseq_input)
    z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense_linear)
    z_mean_encoded = Activation('relu')(z_mean_dense_batchnorm)

    z_log_var_dense_linear = Dense(latent_dim, kernel_initializer='glorot_uniform')(rnaseq_input)
    z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense_linear)
    z_log_var_encoded = Activation('relu')(z_log_var_dense_batchnorm)

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean_encoded, z_log_var_encoded])

    # The decoding layer is much simpler with a single layer and sigmoid activation
    decoder_to_reconstruct = Dense(original_dim, kernel_initializer='glorot_uniform', activation='sigmoid')
    rnaseq_reconstruct = decoder_to_reconstruct(z)

    # Model to compress input
    encoder = Model(rnaseq_input, z_mean_encoded)

    vae_layer = CustomVariationalLayer()([rnaseq_input, rnaseq_reconstruct])
    vae = Model(rnaseq_input, vae_layer)

    adam = optimizers.Adam(lr=learning_rate)
    vae.compile(optimizer=adam, loss=None, loss_weights=[beta])

    return vae, encoder



def get_dumb_autoencoder(original_dim,
                         latent_dim,
                         intermediate_dim):

    def mean_absolute_error(y_true, y_pred):
        return K.mean(K.abs((y_pred - y_true))) / K.mean(y_true)

    def std_absolute_error(y_true, y_pred):
        return K.std(K.abs((y_pred - y_true))) / K.std(y_true)

    input_layer = Input(shape=(original_dim,))
    gn1 = GaussianDropout(0.0)(input_layer)  # I'm not using Gaussian Dropout for now
    encoded = Dense(intermediate_dim, activation='relu')(gn1)
    encoded = Dense(latent_dim, activation='linear')(encoded)
    decoded = Dense(intermediate_dim, activation='relu')(encoded)
    decoded = Dense(original_dim, activation='linear')(decoded)  # this model maps an input to its reconstruction

    autoencoder = Model(input_layer, decoded)

    encoder = Model(input_layer, encoded)
    encoded_input = Input(shape=(latent_dim,))
    decoder_layer1 = autoencoder.layers[-2]
    decoder_layer2 = autoencoder.layers[-1]

    decoder = Model(encoded_input, decoder_layer2(decoder_layer1(encoded_input)))

    sgd = optimizers.SGD(lr=0.1, clipnorm=1.)
    autoencoder.compile(optimizer=sgd, loss='mean_squared_error',
                        metrics=[mean_absolute_error, std_absolute_error])

    return autoencoder, encoder
