"""Provides random number generators for complex data types."""

import logging

import cupy as cp
import numpy as np

import tike.precision

randomizer_np = np.random.default_rng()

logger = logging.getLogger(__name__)


def numpy_complex(*shape):
    """Return a complex random array in the range [-0.5, 0.5)."""
    return (
        randomizer_np.random(size=(*shape, 2), dtype=tike.precision.floating) -
        0.5).view(tike.precision.cfloating)[..., 0]


def cupy_complex(*shape):
    """Return a complex random array in the range [-0.5, 0.5)."""
    return (
        cp.random.random(size=(*shape, 2), dtype=tike.precision.floating) -
        0.5).view(tike.precision.cfloating)[..., 0]


def cluster_wobbly_center(*args, **kwargs):
    """Deprecated alias for :py:func:`tike.cluster.wobbly_center`."""
    import warnings
    warnings.warn(
        'tike.random.cluster_wobbly_center is depreacted. '
        'Use tike.cluster.wobbly_center instead.',
        DeprecationWarning,
    )
    import tike.cluster
    return tike.cluster.wobbly_center(*args, **kwargs)


def cluster_compact(*args, **kwargs):
    """Deprecated alias for :py:func:`tike.cluster.compact`."""
    import warnings
    warnings.warn(
        'tike.random.cluster_compact is depreacted. '
        'Use tike.cluster.compact instead.',
        DeprecationWarning,
    )
    import tike.cluster
    return tike.cluster.compact(*args, **kwargs)
