import tensorflow as tf
import numpy as np
import math

class MultivariateGaussian(object):
    r"""
    Multivariate Gaussian distribution, specified by the mean vector
    and the Cholesky factor of its covariance matrix.

    Parameters
    ----------
    mu
        mean vector, of shape (..., d)
    L
        Lower triangular Cholesky factor of covariance matrix, of shape
        (..., d, d)
    F
        A module that can either refer to the Symbol API or the NDArray
        API in MXNet
    """

    is_reparameterizable = True

    def __init__(
        self, mu, L,  float_type = np.float32
    ) -> None:
        self.mu = mu
        self.L = L
        self.float_type = float_type

    @property
    def batch_shape(self) :
        return self.mu.shape[:-1]

    @property
    def event_shape(self) :
        return self.mu.shape[-1:]

    @property
    def event_dim(self) -> int:
        return 1

    def log_prob(self, x) :
        # todo add an option to compute loss on diagonal covariance only to save time
        print('log prob input  shape', x.shape)
        print('self.mu , self.L  shape', self.mu.shape , self.L.shape)

        # remark we compute d from the tensor but we could ask it to the user alternatively
        d = tf.math.reduce_max(
            tf.math.reduce_sum(tf.ones_like(self.mu), axis=-1)
        )
        residual = tf.expand_dims(x - self.mu ,axis=-1)

        # L^{-1} * (x - mu)
        L_inv_times_residual = tf.linalg.triangular_solve(self.L, residual)
        print("L_inv_times_residual shape",L_inv_times_residual.shape)

        ll = (
            tf.math.subtract(
                -d / 2 * math.log(2 * math.pi)
                ,tf.math.reduce_sum(tf.math.log(tf.linalg.diag_part(self.L))
                                    , axis=1
                )
            )
            - 1
            / 2
            * tf.squeeze(
                tf.linalg.matmul(L_inv_times_residual, tf.transpose(L_inv_times_residual, [0, 2, 1]))
            )
        )

        return ll

    @property
    def mean(self) :
        return self.mu

    @property
    def variance(self) :
        return tf.linalg.matmul(self.L, self.L, transpose_b=True)


        return _sample_multiple(
            s, mu=self.mu, L=self.L, num_samples=num_samples
        )
