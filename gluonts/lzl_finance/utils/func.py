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
        tile_times = [1]*len(param_exp.shape) ; tile_times[axis] = tile_times[axis]*size
        param = tf.tile(param_exp ,tile_times)
        # new_shape = tf.TensorShape(new_shape)
        # param = tf.broadcast_to(param_exp, new_shape )
    return param

# used for the innovation SSM
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

# used for innovation SSM
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



def erf(x):
    # Using numerical recipes approximation for erf function
    # accurate to 1E-7

    ones = tf.ones_like(x)
    zeros = tf.zeros_like(x)
    t = ones / (ones + 0.5 * tf.math.abs(x))

    coefficients = [
        1.00002368,
        0.37409196,
        0.09678418,
        -0.18628806,
        0.27886807,
        -1.13520398,
        1.48851587,
        -0.82215223,
        0.17087277,
    ]

    inner = zeros
    for c in coefficients[::-1]:
        inner = t * (c + inner)

    res = ones - t * tf.math.exp(
        (inner - 1.26551223 - tf.math.square(x))
    )
    return tf.where(tf.math.greater_equal(x, zeros), res, -1.0 * res)

def erfinv(x: Tensor) -> Tensor:
    zeros = tf.zeros_like(x)

    w = -tf.log(tf.math.multiply((1.0 - x), (1.0 + x)))
    mask_lesser = tf.math.less(w, zeros + 5.0)

    w = tf.where(mask_lesser, w - 2.5, tf.math.sqrt(w) - 3.0)

    coefficients_lesser = [
        2.81022636e-08,
        3.43273939e-07,
        -3.5233877e-06,
        -4.39150654e-06,
        0.00021858087,
        -0.00125372503,
        -0.00417768164,
        0.246640727,
        1.50140941,
    ]

    coefficients_greater_equal = [
        -0.000200214257,
        0.000100950558,
        0.00134934322,
        -0.00367342844,
        0.00573950773,
        -0.0076224613,
        0.00943887047,
        1.00167406,
        2.83297682,
    ]

    p = tf.where(
        mask_lesser,
        coefficients_lesser[0] + zeros,
        coefficients_greater_equal[0] + zeros,
    )

    for c_l, c_ge in zip(
        coefficients_lesser[1:], coefficients_greater_equal[1:]
    ):
        c = tf.where(mask_lesser, c_l + zeros, c_ge + zeros)
        p = c + tf.math.multiply(p, w)

    return tf.math.multiply(p, x)

# 产生适用于 tensorflow 的shape list
def mxnet_slice(tensor : Tensor, axis:int, begin:int, end:int):
    begin_list = [0]*len(tensor.shape) ; begin_list[axis] = begin
    end_list = [-1]*len(tensor.shape) ; end_list[axis] = end-begin
    result = tf.slice(tensor, begin=begin_list , size=end_list)
    return result

def mxnet_swapaxes(tensor , dim1, dim2):
    perm = list(range(len(tensor.shape)))
    perm[dim1] = dim2
    perm[dim2] = dim1
    return tf.transpose(tensor, perm)


def batch_diagonal(
    matrix,
    num_data_points = None,
) :
    """
    This function extracts the diagonal of a batch matrix.

    Parameters
    ----------
    matrix
        matrix of shape (batch_size, num_data_points, num_data_points).
    num_data_points
        Number of rows in the kernel_matrix.

    Returns
    -------
    Tensor
        Diagonals of kernel_matrix of shape (batch_size, num_data_points, 1).

    """
    return tf.linalg.matmul(
        tf.math.multiply(
            tf.eye(num_data_points), matrix
        ),
        tf.ones_like(mxnet_slice(matrix, axis=2, begin=0, end=1)),
    )


# noinspection PyMethodOverriding,PyPep8Naming
# gluonts.support.linalg_support
def jitter_cholesky(
    matrix,
    num_data_points = None,
    float_type = np.float64,
    max_iter_jitter: int = 10,
    neg_tol = -1e-8,
    diag_weight = 1e-6,
    increase_jitter = 10,
) :
    """
    This function applies the jitter method.  It iteratively tries to compute the Cholesky decomposition and
    adds a positive tolerance to the diagonal that increases at each iteration until the matrix is positive definite
    or the maximum number of iterations has been reached.

    Parameters
    ----------
    matrix
        Kernel matrix of shape (batch_size, num_data_points, num_data_points).
    num_data_points
        Number of rows in the kernel_matrix.
    ctx
        Determines whether to compute on the cpu or gpu.
    float_type
        Determines whether to use single or double precision.
    max_iter_jitter
        Maximum number of iterations for jitter to iteratively make the matrix positive definite.
    neg_tol
        Parameter in the jitter methods to eliminate eliminate matrices with diagonal elements smaller than this
        when checking if a matrix is positive definite.
    diag_weight
            Multiple of mean of diagonal entries to initialize the jitter.
    increase_jitter
        Each iteration multiply by jitter by this amount
    Returns
    -------
    Optional[Tensor]
        The method either fails to make the matrix positive definite within the maximum number of iterations
        and outputs an error or succeeds and returns the lower triangular Cholesky factor `L`
        of shape (batch_size, num_data_points, num_data_points)
    """
    num_iter = 0
    diag = batch_diagonal(
        matrix, num_data_points
    )  # shape (batch_size, num_data_points, 1)

    diag_mean = tf.expand_dims(
        tf.math.reduce_mean(diag, axis=1),
        axis=2
    )  # shape (batch_size, 1, 1)
    jitter = tf.zeros_like(diag)  # shape (batch_size, num_data_points, 1)
    # Ensure that diagonal entries are numerically non-negative, as defined by neg_tol
    # TODO: Add support for symbolic case: Cannot use < operator with symbolic variables
    whether_positive_definite = tf.math.reduce_sum(
        tf.where(tf.math.less_equal(diag,neg_tol)
                 ,tf.ones_like(diag) ,tf.zeros_like(diag)
        )
    ) > 0

    while num_iter <= max_iter_jitter:
        try:
            L = tf.linalg.cholesky(
                tf.math.add(
                    matrix,
                    tf.math.multiply(
                        tf.eye(num_data_points,  dtype=float_type),
                        jitter,
                    ),
                )
            )#(bs ,num_data_points , num_data_points)
            # gpu will not throw error but will store nans. If nan, L.sum() = nan and
            # L.nansum() computes the sum treating nans as zeros so the error tolerance can be large.
            # for axis = Null, nansum() and sum() will sum over all elements and return scalar array with shape (1,)
            # TODO: 因为tensorflow无法eager到数据，所以assert 和 if 的判断都无用
            # nan_to_zero_L = tf.where(tf.math.is_nan(L) , tf.zeros_like(L) , tf.ones_like(L))
            # assert tf.math.abs(tf.math.reduce_sum(nan_to_zero_L) - tf.math.reduce_sum(L)) <= 1e-1
            return L
        except:
            if num_iter == 0:
                # Initialize the jitter: constant jitter per each batch
                jitter = (
                    tf.math.multiply(diag_mean, tf.ones_like(jitter))
                    * diag_weight
                )
            else:
                jitter = jitter * increase_jitter
        finally:
            num_iter += 1

