import unittest
from parameterized import parameterized
import numpy as np
import scipy.stats
import tensorflow as tf

import sum_of_gaussians as sog


class CalculateMultivariateGaussianTest(unittest.TestCase):
    @parameterized.expand([
        #           x   amp  mean  cov
        ('on_mean', 1., 1., [1.], [[1.]]),
        ('fwhm', np.sqrt(2*np.log(2)), 1., [0.], [[1.]]),
        ('far_away', 10000., 1., [0.], [[1.]]),
        ('small_var', 1., 1., [0.], [[.5]]),
        ('large_var', 1., 1., [0.], [[2.]]),
        ('negative_mean', 1.5, 1., [-2.5], [[3.]]),
        ('amp_double', 1., 2., [0.], [[2.]]),
        ('amp_zero', 1., 0., [0.], [[2.]]),
    ])
    def test_correctness_1d(self, _, x, amp, mean, cov):
        """Compares our implementation against scipy's one as ground truth."""
        value = sog.calculate_multivariate_gaussian([x], amp, mean, cov).numpy()
        std = np.sqrt(cov)[0, 0]
        scipy_value = std * np.sqrt(2*np.pi) * scipy.stats.norm(loc=mean, scale=std).pdf(x)[0]
        self.assertAlmostEqual(amp * scipy_value, value)

    @parameterized.expand([
        #           x        amp  mean      cov matrix
        ('on_mean', [1., 1.], 1., [1., 1.], [[1., 0.], [0., 1.]]),
        ('not_on_mean', [.5, 1.], 1., [0., 1.], [[1., 0.5], [0.5, 1.]]),
        ('far_away', [100., 100.], 1., [0., 0.], [[1., 0.], [0., 1.]]),
        ('small_var', [1., 2.], 1., [0., 0.], [[.1, 0.], [0., .1]]),
        ('large_var', [1., 2.], 1., [0., 0.], [[10., 0.], [0., 10.]]),
        ('diag_var', [1., 2.], 1., [0., 0.], [[1.5, 0.5], [0.5, 1.]]),
        ('negative_mean', [1., 1.], 1., [-1., 0.5], [[2., 1.], [1., 2.]]),
        ('amp_double', [1., 2.], 2., [0., 0.], [[.1, 0.], [0., .1]]),
        ('amp_zero', [1., 2.], 0., [0., 0.], [[.1, 0.], [0., .1]]),
    ])
    def test_correctness_2d(self, _, x, amp, mean, cov):
        """Compares our implementation against scipy's one as ground truth."""
        value = sog.calculate_multivariate_gaussian([x], amp, mean, cov).numpy()[0]
        stat = scipy.stats.multivariate_normal(mean=mean, cov=cov)
        scipy_value = np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov)) * stat.pdf([x])
        self.assertAlmostEqual(scipy_value, value)

    @parameterized.expand([
        ('1d', [1.], [[0.]]),
        ('2d', [1., 1.], [[0., 0.], [0., 0.]]),
        ('2d', [1., 1.], [[1., 0.], [1., 0.]]),
    ])
    def test_raises_error_on_singular_cov_matrix(self, _, mean, cov):
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, 'Input is not invertible'):
            x = [0.] * len(mean)
            _ = sog.calculate_multivariate_gaussian(x, 1., mean, cov)

    def test_return_shape(self):
        x = np.arange(12 * 3, dtype=np.float32).reshape((12, 3))
        for i in (2, 3, 4, 6):
            with self.subTest(i=i):
                _x = x.reshape((i, -1, 3))
                val = sog.calculate_multivariate_gaussian(_x, 1., [0., 0., 0.], np.identity(3, dtype=np.float32))
                self.assertSequenceEqual(_x.shape[:-1], val.shape)
