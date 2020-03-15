from .config import (reload_config)
from .tool_func import (
    weighted_average,
    plot_train_pred,
    plot_train_epoch_loss,
    plot_train_pred_NoSSMnum,
    del_previous_model_params
)
from .func import (
    make_nd_diag
    ,batch_diagonal
    ,jitter_cholesky
    ,_make_2_block_diagonal
    ,_make_block_diagonal
    ,_broadcast_param
    ,erf
)
from .distribution import MultivariateGaussian, Gaussian
__all__ = [
    'plot_train_pred',
    'plot_train_pred_NoSSMnum',
    'plot_train_epoch_loss',
    'del_previous_model_params',
    "reload_config",
    "weighted_average",
    "make_nd_diag",
    "batch_diagonal",
    "jitter_cholesky",
    "_make_2_block_diagonal",
    "_make_block_diagonal",
    "_broadcast_param",
    "MultivariateGaussian",
    "Gaussian",
    "erf"
]

# fix Sphinx issues, see https://bit.ly/2K2eptM
for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)