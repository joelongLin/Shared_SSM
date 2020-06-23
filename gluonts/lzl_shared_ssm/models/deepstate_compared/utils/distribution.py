import tensorflow as tf
import numpy as np
import math
from .func import erf , erfinv
import tensorflow_probability as tfp

def _expand_param(p, num_samples= None):
    """
    Expand parameters by num_samples along the first dimension.
    """
    if num_samples is None:
        return p
    exp_result = tf.tile(
        tf.expand_dims(p, axis=0)
        ,multiples=[num_samples]+[1]*(len(tf.expand_dims(p, axis=0).shape)-1))
    #(num_sample, (original shape)))
    return exp_result


def _sample_normal_multiple(
     *args, num_samples = None ,**kwargs
) :
    """
    Sample from the sample_func, by passing expanded args and kwargs and
    reshaping the returned samples afterwards.
    """

    args_expanded = [_expand_param(a, num_samples) for a in args]
    kwargs_expanded = {
        k: _expand_param(v, num_samples) for k, v in kwargs.items()
    }

    sample_func = tfp.distributions.Sample(
        distribution=tfp.distributions.Normal(*args_expanded,**kwargs_expanded),
    ).sample

    samples = sample_func() #(num_sample, bs, pred_length, obs_dim, obs_dim)
    return samples

def _sample_multiple(
    sample_func, *args, num_samples = None, **kwargs
) :
    """
    Sample from the sample_func, by passing expanded args and kwargs and
    reshaping the returned samples afterwards.
    """
    args_expanded = [_expand_param(a, num_samples) for a in args]
    kwargs_expanded = {
        k: _expand_param(v, num_samples) for k, v in kwargs.items()
    }


    samples = sample_func(*args_expanded, **kwargs_expanded)#(num_sample, bs, pred_length, obs_dim, obs_dim)
    return samples

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
        # print('log prob input  shape', x.shape)
        # print('self.mu , self.L  shape', self.mu.shape , self.L.shape)

        # remark we compute d from the tensor but we could ask it to the user alternatively
        d = tf.math.reduce_max(
            tf.math.reduce_sum(tf.ones_like(self.mu), axis=-1)
        )
        residual = tf.expand_dims(x - self.mu ,axis=-1)

        # L^{-1} * (x - mu)
        L_inv_times_residual = tf.linalg.triangular_solve(self.L, residual)
        # print("L_inv_times_residual shape",L_inv_times_residual.shape)

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

    def sample(self, num_samples = None) :
        r"""
        Draw samples from the multivariate Gaussian distributions.
        Internally, Cholesky factorization of the covariance matrix is used:

            sample = L v + mu,

        where L is the Cholesky factor, v is a standard normal sample.

        Parameters
        ----------
        num_samples
            Number of samples to be drawn.
        Returns
        -------
        Tensor
            Tensor with shape (num_samples, ..., d).
        """

        def s(mu, L) :
            samples_std_normal = tf.expand_dims(
                tfp.distributions.Sample(
                    tfp.distributions.Normal(
                        loc=tf.zeros_like(mu),
                        scale=tf.ones_like(mu),
                    )
                ).sample(), axis=-1)#(num_sample, latent_dim, latent_dim, 1)
            samples = (
                tf.squeeze(
                    tf.linalg.matmul(L, samples_std_normal),
                    axis=-1
                )
                + mu
            )#(num_sample, latent_dim, latent_dim)
            return samples

        return _sample_multiple(
            s, mu=self.mu, L=self.L, num_samples=num_samples
        )



class Gaussian(object):
    r"""
    Gaussian distribution.

    Parameters
    ----------
    mu
        Tensor containing the means, of shape `(*batch_shape, *event_shape)`.
    std
        Tensor containing the standard deviations, of shape
        `(*batch_shape, *event_shape)`.
    F
    """

    is_reparameterizable = True

    def __init__(self, mu, sigma) -> None:
        self.mu = mu
        self.sigma = sigma

    @property
    def batch_shape(self) :
        return self.mu.shape

    @property
    def event_shape(self) :
        return ()

    @property
    def event_dim(self) :
        return 0

    def log_prob(self, x) :
        mu, sigma = self.mu, self.sigma
        return -1.0 * (
            tf.math.log(sigma)
            + 0.5 * math.log(2 * math.pi)
            + 0.5 * tf.math.square((x - mu) / sigma)
        )

    @property
    def mean(self) :
        return self.mu

    @property
    def stddev(self) :
        return self.sigma

    def cdf(self, x):
        u = tf.math.divide(
            tf.math.subtract(x, self.mu), self.sigma * math.sqrt(2.0)
        )
        return (erf(u) + 1.0) / 2.0

    def sample(self, num_samples = None) :
        return _sample_normal_multiple(
            loc=self.mu,
            scale=self.sigma,
            num_samples=num_samples,
        )

    def sample_rep(self, num_samples = None) :
        def s(mu, sigma):
            raw_samples = self.F.sample_normal(
                mu=mu.zeros_like(), sigma=sigma.ones_like()
            )
            return sigma * raw_samples + mu

        return _sample_normal_multiple(
            s, mu=self.mu, sigma=self.sigma, num_samples=num_samples
        )

    def quantile(self, level) :
        # we consider level to be an independent axis and so expand it
        # to shape (num_levels, 1, 1, ...)
        for _ in range(self.all_dim):
            level = level.expand_dims(axis=-1)

        return tf.math.broadcast_add(
            self.mu,
            tf.math.multiply(
                self.sigma, math.sqrt(2.0) * erfinv(F, 2.0 * level - 1.0)
            ),
        )

