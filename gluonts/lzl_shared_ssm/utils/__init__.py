from .config import (reload_config)
from .tool_func import (
    time_format_from_frequency_str,
    create_dataset_if_not_exist,
    weighted_average,
    plot_train_pred,
    plot_train_epoch_loss,
    plot_train_pred_NoSSMnum,
    del_previous_model_params,
    get_model_params_name,
    complete_batch,
    add_time_mark_to_file,
    add_time_mark_to_dir,
    samples_with_mean_cov,
    get_reload_hyper
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
    'time_format_from_frequency_str',
    'create_dataset_if_not_exist',
    'complete_batch',
    'plot_train_pred',
    'plot_train_pred_NoSSMnum',
    'plot_train_epoch_loss',
    'del_previous_model_params',
    'get_model_params_name',
    'add_time_mark_to_file',
    'add_time_mark_to_dir',
    'samples_with_mean_cov',
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
    "erf",
    "get_reload_hyper"
]

# fix Sphinx issues, see https://bit.ly/2K2eptM
for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)