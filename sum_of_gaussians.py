# For a more detailed explanation on the SumOfGaussian model, please
# refer to: http://zachmoshe.com

import tensorflow as tf
import numpy as np


@tf.function()
def calculate_multivariate_gaussian(positions, amp, mu, sigma):
    """Calculates the multivariate Gaussian distribution on given positions in D dimensions.

    Based on: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

    Args:
      positions: Array of (<inner_shape>, D) of data points of dimension D.
      amp: The amplitude (scalar).
      mu: Array of (D,) with the mean vector.
      sigma: Array of (D,D) with the covariance matrix.
    Returns:
      Array of (<inner_shape>) with the gaussian function value multiplied by the amplitude
      at each point.
    """
    inv_sigma = tf.linalg.inv(sigma)
    flatten_pos = tf.reshape(positions, (-1, tf.shape(positions)[-1]))
    flatten_result = (amp * tf.exp(-0.5 * tf.einsum('ak,kl,al->a', flatten_pos - mu, inv_sigma, flatten_pos - mu)))
    return tf.reshape(flatten_result, tf.shape(positions)[:-1])


def _means_spread_regularizer(means, eps=1e-9):
    """Defines a loss term that penalize for two centers that are close to each other."""
    num_means = means.shape[0]

    means_pairwise_distances = tf.norm(tf.expand_dims(means, 0) - tf.expand_dims(means, 1) + eps, axis=-1)
    matrix_upper_half_only = tf.linalg.band_part(means_pairwise_distances, 0, -1)
    num_distances = num_means * (num_means - 1) / 2
    avg_distance = tf.reduce_sum(matrix_upper_half_only) / num_distances

    return 1. / avg_distance


def _multiple_identity_matrices_initializer(shape, dtype=None):
    """Initialize a stacked tensor of multiple identity matrices."""
    del dtype  # unused.
    assert len(shape) == 3 and shape[1] == shape[2], 'shape must be (N, D, D)'
    return np.stack([np.identity(shape[1]) for _ in range(shape[0])])


@tf.function()
def _pseudo_sigma_to_sigma(pseudo_sigma):
    """Returns a legal non-singular covariance matrix from a "pseudo_sigma" tensor."""
    pseudo_sigma = tf.linalg.band_part(pseudo_sigma, 0, -1)  # take only upper half.
    sigma = pseudo_sigma @ tf.transpose(pseudo_sigma)

    # make sure it's non singular
    sigma += tf.cast(tf.identity(sigma.shape[0]), sigma.dtype) * 1e-6

    return sigma


class SumOfGaussians(tf.keras.layers.Layer):
    """Assumes data is a sum of multiple Gaussians and learns the means, covariances and amplitudes."""
    def __init__(self, num_gaussians, amps_l1_reg=1e-3, use_means_spread_regularizer=True,
                 force_amps_nonneg=True, centers_min=0., centers_max=1.,
                 **kwargs):
        super(SumOfGaussians, self).__init__(**kwargs)
        self.num_gaussians = num_gaussians
        self.amps_regularizer = tf.keras.regularizers.l1(amps_l1_reg)
        self.amps_constraint = (tf.keras.constraints.NonNeg()
                                if force_amps_nonneg
                                else None)
        self.centers_regularizer = (_means_spread_regularizer
                                    if use_means_spread_regularizer
                                    else None)

        self.centers_min = np.array(centers_min)
        self.centers_max = np.array(centers_max)

    def build(self, input_shape):
        self.dim = input_shape[-1]
        if (self.centers_min.shape != self.centers_max.shape) or (self.centers_min.shape not in ((), (self.dim,))):
            raise ValueError('Illegal centers_min, centers_max. Must be either a number or a (dim,) shape array')

        self.centers_initializer = tf.keras.initializers.RandomUniform(
            minval=self.centers_min, maxval=self.centers_max)

        self.means = self.add_weight(
            'means', shape=(self.num_gaussians, self.dim), dtype=tf.float32,
            initializer=self.centers_initializer,
            regularizer=self.centers_regularizer)

        self.pseudo_sigmas = self.add_weight(
            'pseudo_sigmas', shape=(self.num_gaussians, self.dim, self.dim), dtype=tf.float32,
            initializer=_multiple_identity_matrices_initializer)

        self.amps = self.add_weight(
            'amps', shape=(self.num_gaussians,), dtype=tf.float32,
            initializer=tf.keras.initializers.Ones(),
            regularizer=self.amps_regularizer,
            constraint=self.amps_constraint)

    def _get_sigma(self, i):
        return _pseudo_sigma_to_sigma(self.pseudo_sigmas[i])

    def get_current_amps_regularizer_value(self):
        return self.amps_regularizer(self.amps)

    def call(self, inputs, **kwargs):
        del kwargs  # unused.
        per_gaussian_output = [
            calculate_multivariate_gaussian(inputs, self.amps[i], self.means[i], self._get_sigma(i))
            for i in range(self.num_gaussians)]
        result = tf.reduce_sum(tf.stack(per_gaussian_output, axis=-1), axis=-1, keepdims=True)
        return result
