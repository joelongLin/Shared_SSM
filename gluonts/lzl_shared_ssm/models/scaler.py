# -*- coding: UTF-8 -*-
# author : joelonglin
import numpy as np
import tensorflow as tf

class Scaler(object):
    """
    Base class for blocks used to scale data.
    Parameters
    ----------
    keepdims
        toggle to keep the dimension of the input tensor.
    """

    def __init__(self, keepdims = False):
        self.keepdims = keepdims

    def compute_scale(self, data, observed_indicator, seq_axis):
        """
        Computes the scale of the given input data.
        Parameters
        ----------
        data
            数据
        observed_indicator
            观测值
        seq_axis
            指定seq 所在的 维度
        """
        raise NotImplementedError()

    # noinspection PyMethodOverriding
    def build_forward(
        self, data, observed_indicator, seq_axis
    ) :
        """
        Parameters
        ----------
            与 compute_scale一样
        Returns
        -------
        Tensor
            被scaled之后的值
        Tensor
            scaled 函数是否要保持维度一致

        """
        scale = self.compute_scale(data, observed_indicator, seq_axis)

        if self.keepdims:
            scale = tf.expand_dims(scale,axis=seq_axis)
            return tf.math.divide(data, scale), scale
        else:
            return tf.math.divide(data, tf.expand_dims(scale,axis=seq_axis)), scale

class MeanScaler(Scaler):
    """
    The ``MeanScaler`` computes a per-item scale according to the average
    absolute value over time of each item. The average is computed only among
    the observed values in the data tensor, as indicated by the second
    argument. Items with no observed data are assigned a scale based on the
    global average.

    Parameters
    ----------
    minimum_scale
        default scale that is used if the time series has only zeros.
    """

    def __init__(self, minimum_scale: float = 1e-10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.minimum_scale = minimum_scale

    def compute_scale(
        self, data, observed_indicator,seq_axis  # shapes (N, T, C)
    ) :
        """
        Parameters
        ----------
        data
            tensor of shape (..., N, T, C) containing the data to be scaled

        observed_indicator
            observed_indicator: binary tensor with the same shape as
            ``data``, that has 1 in correspondence of observed data points,
            and 0 in correspondence of missing data points.

        Returns
        -------
        Tensor
            shape (N, C), computed according to the
            average absolute value over time of the observed values.
        """
        # these will have shape (...,N, C) 注意：这里的sample
        num_observed = tf.math.reduce_sum(observed_indicator, axis=seq_axis) # 计算每一个sample中，观测到 数据 的数量
        sum_observed = tf.math.reduce_sum((tf.math.abs(data) * observed_indicator),axis=seq_axis) # 计每一个sample中，观测值的绝对值的和

        # first compute a global scale per-dimension
        total_observed = tf.math.reduce_sum(num_observed,axis=seq_axis-1) #计算一个batch里面，观测到 数据 的数量
        denominator = tf.math.maximum(total_observed, 1.0)
        # shape (...,1,C, ) # batch观测到的值的绝对值的和 / batch观测到的值的数量
        default_scale = tf.math.reduce_sum(sum_observed,axis=seq_axis-1) / denominator
        default_scale = tf.expand_dims(default_scale, axis=seq_axis-1)

        # then compute a per-item, per-dimension scale
        denominator = tf.math.maximum(num_observed, 1.0)
        # sample观测到的值的绝对值的和 / sample观测到的值的数量
        scale = sum_observed / denominator  # shape (..., N, C)

        # use per-batch scale when no element is observed
        # or when the sequence contains only zeros
        cond = tf.math.greater(sum_observed, tf.zeros_like(sum_observed))
        scale = tf.where(
            cond,
            scale,
            tf.math.multiply(default_scale, tf.ones_like(num_observed)),
        )

        return tf.math.maximum(scale, self.minimum_scale)


class NOPScaler(Scaler):
    """
    The ``NOPScaler`` assigns a scale equals to 1 to each input item, i.e.,
    no scaling is applied upon calling the ``NOPScaler``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # noinspection PyMethodOverriding
    def compute_scale(
        self, F, data, observed_indicator
    ) :
        """
        Parameters
        ----------
        F
            A module that can either refer to the Symbol API or the NDArray
            API in MXNet.

        data
            tensor of shape (N, T, C) containing the data to be scaled

        observed_indicator
            observed_indicator: binary tensor with the same shape as
            ``data``, that has 1 in correspondence of observed data points,
            and 0 in correspondence of missing data points.

        Returns
        -------
        Tensor
            shape (N, C), identically equal to 1.
        """
        return tf.math.reduce_mean(tf.ones_like(data),axis=1)