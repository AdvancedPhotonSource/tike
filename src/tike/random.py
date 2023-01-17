"""Provides random number generators for complex data types."""

import logging

import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)


def numpy_complex(*shape):
    """Return a complex random array in the range [-0.5, 0.5)."""
    return (np.random.rand(*shape, 2) - 0.5).view('complex')[..., 0]


def cupy_complex(*shape):
    """Return a complex random array in the range [-0.5, 0.5)."""
    return (cp.random.rand(*shape, 2) - 0.5).view('complex')[..., 0]


def cluster_wobbly_center(*args, **kwargs):
    import warnings
    warnings.warn(
        'tike.random.cluster_wobbly_center is depreacted. '
        'Use tike.cluster.wobbly_center instead.',
        DeprecationWarning,
    )
    import tike.cluster
    return tike.cluster.wobbly_center(*args, **kwargs)


def cluster_compact(*args, **kwargs):
    import warnings
    warnings.warn(
        'tike.random.cluster_compact is depreacted. '
        'Use tike.cluster.compact instead.',
        DeprecationWarning,
    )
    import tike.cluster
    return tike.cluster.compact(*args, **kwargs)
