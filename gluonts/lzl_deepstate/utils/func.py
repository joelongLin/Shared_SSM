import tensorflow as tf
from tensorflow import Tensor
import numpy as np

def make_nd_diag(x, d) :
    """
    Make a diagonal tensor, given the diagonal
    """
    return tf.multiply(tf.eye(d), tf.expand_dims(x ,axis=-1))

#finish
def _broadcast_param(param, axes, sizes):
    for axis, size in zip(axes, sizes):
        param_exp = tf.expand_dims(param,axis = axis)
        new_shape = list(param_exp.shape) ; new_shape[-1] = new_shape[-1]*size
        param = tf.broadcast_to(param_exp, new_shape)
    return param

def _make_block_diagonal(blocks):
    assert (
        len(blocks) > 0
    ), "You need at least one tensor to make a block-diagonal tensor"

    if len(blocks) == 1:
        return blocks[0]


    # transition coefficient is block diagonal!
    block_diagonal = _make_2_block_diagonal(blocks[0], blocks[1])
    for i in range(2, len(blocks)):
        block_diagonal = _make_2_block_diagonal(
             left=block_diagonal, right=blocks[i]
        )

    return block_diagonal


def _make_2_block_diagonal(left, right) :
    """
    Creates a block diagonal matrix of shape (batch_size, m+n, m+n) where m and n are the sizes of
    the axis 1 of left and right respectively.

    Parameters
    ----------
    tf
    left
        Tensor of shape (batch_size, seq_length, m, m)
    right
        Tensor of shape (batch_size, seq_length, n, n)
    Returns
    -------
    Tensor
        Block diagonal matrix of shape (batch_size, seq_length, m+n, m+n)
    """
    # shape (batch_size, seq_length, m, n)
    begin = [0]*len(left.shape)
    left_size = [-1]*len(left.shape);left_size[-1] =1
    right_size= [-1]*len(right.shape) ; right_size[-2] =1
    zeros_off_diag = tf.add(
        tf.zeros_like(tf.slice(left,begin=begin ,size=left_size))
        ,  # shape (batch_size, seq_length, m, 1)
        tf.zeros_like(tf.slice(right, begin=begin, size=right_size))
        ,  # shape (batch_size, seq_length, 1, n)
    )#shape (batch_size, m, n)

    # shape (batch_size, n, m)
    zeros_off_diag_tr = tf.transpose(zeros_off_diag,[0,1,3,2])

    # block diagonal: shape (batch_size, seq_length, m+n, m+n)
    _block_diagonal = tf.concat(
        [tf.concat([left, zeros_off_diag], axis=3),  #shape (batch_size, seq_length, m, m+n)
        tf.concat([zeros_off_diag_tr, right], axis=3)], #shape (batch_size, seq_length, n, m+n)
        axis=2,
    )

    return _block_diagonal

def weighted_average(
   x, weights = None, axis=None
):
    """
    Computes the weighted average of a given tensor across a given axis.
    Parameters
    ----------
    x
        Input tensor, of which the average must be computed. #(bs,seq_length )
    weights
        Weights tensor, of the same shape as `x`.  #(bs,seq_length )
    axis
        The axis along which to average `x`

    Returns
    -------
    Tensor:
        The tensor with values averaged along the specified `axis`.
    """


    if weights is not None:
        weighted_tensor = x * weights
        sum_weights = tf.math.maximum(1.0, tf.math.reduce_sum(weights,axis=axis))
        return tf.math.reduce_sum(weighted_tensor,axis=axis) / sum_weights
    else:
        return tf.math.reduce_mean(x ,axis=axis)
