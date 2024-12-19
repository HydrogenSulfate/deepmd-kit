# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import os
import random

import numpy as np
import paddle

from deepmd.common import (
    VALID_PRECISION,
)
from deepmd.env import (
    GLOBAL_ENER_FLOAT_PRECISION,
    GLOBAL_NP_FLOAT_PRECISION,
    get_default_nthreads,
    set_default_nthreads,
)

log = logging.getLogger(__name__)

SAMPLER_RECORD = os.environ.get("SAMPLER_RECORD", False)
DP_DTYPE_PROMOTION_STRICT = os.environ.get("DP_DTYPE_PROMOTION_STRICT", "0") == "1"
try:
    # only linux
    ncpus = len(os.sched_getaffinity(0))
except AttributeError:
    ncpus = os.cpu_count()
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", min(0, ncpus)))
# Make sure DDP uses correct device if applicable
LOCAL_RANK = paddle.distributed.get_rank()

if os.environ.get("DEVICE") == "cpu" or paddle.device.cuda.device_count() <= 0:
    DEVICE = "cpu"
else:
    DEVICE = f"gpu:{LOCAL_RANK}"

paddle.device.set_device(DEVICE)

JIT = bool(int(os.environ.get("JIT", False)))
CINN = bool(int(os.environ.get("CINN", False)))
CACHE_PER_SYS = 5  # keep at most so many sets per sys in memory
ENERGY_BIAS_TRAINABLE = True

PRECISION_DICT = {
    "float16": paddle.float16,
    "float32": paddle.float32,
    "float64": paddle.float64,
    "half": paddle.float16,
    "single": paddle.float32,
    "double": paddle.float64,
    "int32": paddle.int32,
    "int64": paddle.int64,
    "bfloat16": paddle.bfloat16,
    "bool": paddle.bool,
}
GLOBAL_PD_FLOAT_PRECISION = PRECISION_DICT[np.dtype(GLOBAL_NP_FLOAT_PRECISION).name]
GLOBAL_PD_ENER_FLOAT_PRECISION = PRECISION_DICT[
    np.dtype(GLOBAL_ENER_FLOAT_PRECISION).name
]
PRECISION_DICT["default"] = GLOBAL_PD_FLOAT_PRECISION
assert VALID_PRECISION.issubset(PRECISION_DICT.keys())
# cannot automatically generated
RESERVED_PRECISON_DICT = {
    paddle.float16: "float16",
    paddle.float32: "float32",
    paddle.float64: "float64",
    paddle.int32: "int32",
    paddle.int64: "int64",
    paddle.bfloat16: "bfloat16",
    paddle.bool: "bool",
}
assert set(PRECISION_DICT.values()) == set(RESERVED_PRECISON_DICT.keys())
DEFAULT_PRECISION = "float64"

# throw warnings if threads not set
set_default_nthreads()
inter_nthreads, intra_nthreads = get_default_nthreads()
# if inter_nthreads > 0:  # the behavior of 0 is not documented
#     os.environ['OMP_NUM_THREADS'] = str(inter_nthreads)
# if intra_nthreads > 0:
#     os.environ['CPU_NUM'] = str(intra_nthreads)


def enable_prim(enable: bool = True):
    # NOTE: operator in list below will not use composite
    # operator but kernel instead
    EAGER_COMP_OP_BLACK_LIST = [
        "abs_grad",
        "cast_grad",
        # "concat_grad",
        "cos_double_grad",
        "cos_grad",
        "cumprod_grad",
        "cumsum_grad",
        "dropout_grad",
        "erf_grad",
        "exp_grad",
        "expand_grad",
        "floor_grad",
        "gather_grad",
        "gather_nd_grad",
        "gelu_grad",
        "group_norm_grad",
        "instance_norm_grad",
        "layer_norm_grad",
        "leaky_relu_grad",
        "log_grad",
        "max_grad",
        "pad_grad",
        "pow_double_grad",
        "pow_grad",
        "prod_grad",
        "relu_grad",
        "roll_grad",
        "rsqrt_grad",
        "scatter_grad",
        "scatter_nd_add_grad",
        "sigmoid_grad",
        "silu_grad",
        "sin_double_grad",
        "sin_grad",
        "slice_grad",
        # "split_grad",
        "sqrt_grad",
        "stack_grad",
        "sum_grad",
        "tanh_double_grad",
        "tanh_grad",
        "topk_grad",
        "transpose_grad",
        "add_double_grad",
        "add_grad",
        "assign_grad",
        "batch_norm_grad",
        "divide_grad",
        "elementwise_pow_grad",
        "maximum_grad",
        "min_grad",
        "minimum_grad",
        "multiply_grad",
        "subtract_grad",
        "tile_grad",
    ]
    EAGER_COMP_OP_BLACK_LIST = list(set(EAGER_COMP_OP_BLACK_LIST))
    # NOTE: operator in list below will not use composite
    # operator but kernel instead
    STATIC_COMP_OP_BLACK_LIST = [
        ##### forward decomp #####
        "pd_op.any",
        "pd_op.mean",
        "pd_op.p_norm",
        "pd_op.pow",
        "pd_op.one_hot",
        "pd_op.squared_l2_norm",
        "pd_op.reciprocal",
        "pd_op.bce_loss",
        "pd_op.bmm",
        "pd_op.softmax",
        "pd_op.log_softmax",
        "pd_op.stack",
        "pd_op.silu",
        "pd_op.swiglu",
        "pd_op.relu",
        "pd_op.relu6",
        "pd_op.squeeze",
        "pd_op.unsqueeze",
        "pd_op.add_n",
        "pd_op.full_like",
        "pd_op.gelu",
        "pd_op.hardsigmoid",
        "pd_op.hardswish",
        "pd_op.heaviside",
        "pd_op.leaky_relu",
        "pd_op.flatten",
        "pd_op.clip",
        "pd_op.index_select",
        "pd_op.square",
        "pd_op.sigmoid_cross_entropy_with_logits",
        "pd_op.mean_all",
        "pd_op.embedding",
        "pd_op.index_sample",
        "pd_op.elu",
        "pd_op.lerp",
        "pd_op.log_loss",
        "pd_op.kldiv_loss",
        "pd_op.softsign",
        "pd_op.numel",
        "pd_op.swish",
        "pd_op.addmm",
        ##### backward decomp #####
        # "pd_op.abs_grad",
        "pd_op.assign_grad",
        "pd_op.bce_loss_grad",
        "pd_op.cumsum_grad",
        "pd_op.cumprod_grad",
        "pd_op.divide_grad",
        "pd_op.floor_grad",
        "pd_op.sum_grad",
        "pd_op.mean_grad",
        "pd_op.gelu_grad",
        "pd_op.reduce_as_grad",
        # "pd_op.reshape_grad",
        "pd_op.roll_grad",
        "pd_op.transpose_grad",
        "pd_op.scatter_grad",
        "pd_op.scatter_nd_add_grad",
        "pd_op.sin_grad",
        "pd_op.cos_grad",
        # "pd_op.tanh_grad",
        "pd_op.concat_grad",
        "pd_op.split_grad",
        "pd_op.cast_grad",
        # "pd_op.add_grad",
        # "pd_op.subtract_grad",
        "pd_op.multiply_grad",
        "pd_op.elementwise_pow_grad",
        "pd_op.pow_grad",
        "pd_op.scale_grad",
        "pd_op.stack_grad",
        "pd_op.layer_norm_grad",
        "pd_op.dropout_grad",
        "pd_op.erf_grad",
        "pd_op.expand_grad",
        # "pd_op.log_grad",
        "pd_op.square_grad",
        # "pd_op.exp_grad",
        "pd_op.sqrt_grad",
        "pd_op.rsqrt_grad",
        # "pd_op.silu_grad",
        # "pd_op.softmax_grad",
        "pd_op.squeeze_grad",
        "pd_op.unsqueeze_grad",
        # "pd_op.matmul_grad",
        # "pd_op.maximum_grad",
        "pd_op.masked_select_grad",
        "pd_op.relu_grad",
        "pd_op.gather_grad",
        # "pd_op.gather_nd_grad",
        "pd_op.instance_norm_grad",
        "pd_op.pad_grad",
        "pd_op.max_grad",
        "pd_op.slice_grad",
        "pd_op.tile_grad",
        "pd_op.hardswish_grad",
        "pd_op.leaky_relu_grad",
        "pd_op.sigmoid_grad",
        "pd_op.topk_grad",
        "pd_op.batch_norm_grad",
        "pd_op.prod_grad",
        # "pd_op.minimum_grad",
        "pd_op.group_norm_grad",
        "pd_op.swiglu_grad",
        "pd_op.softsign_grad",
        # "pd_op.where_grad",
        "pd_op.expm1_grad",
        "pd_op.atan2_grad",
        "pd_op.put_along_axis_grad",
        "pd_op.atan_grad",
        "pd_op.swish_grad",
        "pd_op.fmax_grad",
        "pd_op.fmin_grad",
        "pd_op.dot_grad",
        "pd_op.logcumsumexp_grad",
        "pd_op.logsumexp_grad",
        "pd_op.trunc_grad",
        "pd_op.kthvalue_grad",
        "pd_op.argsort_grad",
        "pd_op.kron_grad",
        # "pd_op.take_along_axis_grad",
        "pd_op.ceil_grad",
        "pd_op.amax_grad",
        "pd_op.amin_grad",
        # "pd_op.p_norm_grad",
    ]
    """Enable running program in primitive C++ API in eager/static mode."""
    from paddle.framework import (
        core,
    )

    core.set_prim_eager_enabled(enable)
    paddle.framework.core._set_prim_all_enabled(enable)
    if JIT is False and enable:
        paddle.framework.core._set_prim_backward_blacklist(*EAGER_COMP_OP_BLACK_LIST)
    if JIT and enable:
        paddle.framework.core._set_prim_backward_blacklist(*STATIC_COMP_OP_BLACK_LIST)
        log.info(f"Add {STATIC_COMP_OP_BLACK_LIST} into static decomposite blacklist")
    log.info(
        f"******************** {'Enable' if enable else 'Disable'} prim in eager and static mode.********************"
    )


def set_random_seed(seed: int):
    """Set random seed."""
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    log = logging.getLogger(__name__)
    log.info("-" * 20 + f"Set random seed to {seed}" + "-" * 20)


__all__ = [
    "CACHE_PER_SYS",
    "DEFAULT_PRECISION",
    "DEVICE",
    "ENERGY_BIAS_TRAINABLE",
    "GLOBAL_ENER_FLOAT_PRECISION",
    "GLOBAL_NP_FLOAT_PRECISION",
    "GLOBAL_PD_ENER_FLOAT_PRECISION",
    "GLOBAL_PD_FLOAT_PRECISION",
    "JIT",
    "LOCAL_RANK",
    "NUM_WORKERS",
    "PRECISION_DICT",
    "RESERVED_PRECISON_DICT",
    "SAMPLER_RECORD",
    "enable_prim",
    "set_random_seed",
]
