"""Provides random number generators for complex data types."""

import cupy as cp
import numpy as np

randomizer = np.random.default_rng()

def numpy_complex(*shape):
    """Return a complex random array in the range [-0.5, 0.5)."""
    return (randomizer.random(*shape, 2) - 0.5).view('complex')[..., 0]


def cupy_complex(*shape):
    """Return a complex random array in the range [-0.5, 0.5)."""
    return (cp.random.rand(*shape, 2) - 0.5).view('complex')[..., 0]
