"""Functions related to creating and manipulating probe arrays.

Ptychographic objects are stored as a single complex array.

"""

import logging

import cupy as cp
import cupyx.scipy.ndimage

logger = logging.getLogger(__name__)


# TODO: Use dataclass decorator when python 3.6 reaches EOL
class ObjectOptions:
    """Manage data and setting related to object correction."""

    def __init__(self, positivity_constraint=0, smoothness_constraint=0):
        self.positivity_constraint = positivity_constraint
        self.smoothness_constraint = positivity_constraint


def positivity_constraint(x, r):
    """Constrains the amplitude of x to be positive with sum of abs(x) and x.

    Parameters
    ----------
    r : float [0, 1]
        The proportion of abs(x) in the weighted sum of abs(x) and x.

    """
    if r > 0:
        if r > 1:
            raise ValueError(
                f"Positivity constraint must be in the range [0, 1] not {r}.")
        logger.info("Object positivity constrained with ratio %.3e", r)
        return r * cp.abs(x) + (1 - r) * x
    else:
        return x


def smoothness_constraint(x, a):
    """Convolves the image with a 3x3 averaging kernel.

    The kernel is defined as::

        [[a, a, a]
         [a, c, a]
         [a, a, a]]

    where c = 1 - 8 * a

    Parameters
    ----------
    a : float [0, 1/8)
        The non-center weights of the kernel.
    """
    if 0 <= a and a < 1.0 / 8.0:
        logger.info("Object smooth constrained with kernel param %.3e", a)
        weights = cp.ones([1] * (x.ndim - 2) + [3, 3], dtype='float32') * a
        weights[..., 1, 1] = 1.0 - 8.0 * a
        x.real = cupyx.scipy.ndimage.convolve(x.real, weights, mode='nearest')
        x.imag = cupyx.scipy.ndimage.convolve(x.imag, weights, mode='nearest')
        return x
    else:
        raise ValueError(
            f"Smoothness constraint must be in range [0, 1/8) not {a}.")
