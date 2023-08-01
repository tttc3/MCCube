from typing import Tuple

import equinox as eqx
import numpy as np
from jaxtyping import ArrayLike, Float


class ResultStatistics(eqx.Module):
    value: ArrayLike
    maximum: float
    minimum: float
    dimension_averaged: ArrayLike
    l2: float


def _error_stats(a, b):
    abserr = np.abs(a - b)
    l2err = np.linalg.norm(a - b)
    return ResultStatistics(a, np.max(abserr), np.min(abserr), np.mean(abserr), l2err)


def cubature_target_error(
    particles: Float[ArrayLike, "n d"],
    target_mean: Float[ArrayLike, " d"],
    target_covariance: Float[ArrayLike, "d d"],
) -> Tuple[ResultStatistics, ResultStatistics]:
    mean_err = _error_stats(np.mean(particles, axis=0), target_mean)
    cov_err = _error_stats(np.cov(particles, rowvar=False), target_covariance)
    return (mean_err, cov_err)
