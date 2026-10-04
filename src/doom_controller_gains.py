"""Project a three-dimensional search onto a frozen reference controller."""

import numpy as np


FEATURE_BLOCK_SIZES = (64, 512, 512)  # z, cell, hidden: original control order.
MIN_GAIN = 0.25
MAX_GAIN = 4.0


def controller_from_log_gains(initial, log_gains):
    """Scale feature blocks once on CPU; original singleton inference is unchanged.

    Zero log gains reproduce the initializer's raw FP64 weights exactly. This
    changes the controller, including the raw action fed back to its fixed RNN;
    it is not equivalent to changing only the environment's button threshold.
    """
    initial = np.asarray(initial)
    if (
        initial.shape != (sum(FEATURE_BLOCK_SIZES),)
        or initial.dtype != np.float64
        or not np.isfinite(initial).all()
    ):
        raise ValueError("Expected 1088 finite FP64 frozen controller weights")
    log_gains = np.asarray(log_gains, dtype=np.float64)
    if (
        log_gains.shape != (len(FEATURE_BLOCK_SIZES),)
        or not np.isfinite(log_gains).all()
        or (log_gains < np.log(MIN_GAIN)).any()
        or (log_gains > np.log(MAX_GAIN)).any()
    ):
        raise ValueError("Expected three finite log gains within [log(.25), log(4)]")
    scales = np.repeat(np.exp(log_gains), FEATURE_BLOCK_SIZES)
    with np.errstate(over="ignore", invalid="ignore"):
        parameters = initial * scales
    if not np.isfinite(parameters).all():
        raise ValueError("Projected controller contains non-finite weights")
    return parameters
