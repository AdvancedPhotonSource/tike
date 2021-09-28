"""Functions related to creating and manipulating ptychographic object arrays.

Ptychographic objects are stored as a single complex array which represent
the complex refractive indices in the field of view.

"""

import logging

import cupy as cp
import cupyx.scipy.ndimage
import numpy as np

logger = logging.getLogger(__name__)


# TODO: Use dataclass decorator when python 3.6 reaches EOL
class ObjectOptions:
    """Manage data and setting related to object correction.

    Attributes
    ----------
    positivity_constraint : float [0, 1]
        This value is passed to the tike.ptycho.object.positivity_constraint
        function.
    smoothness_constraint : float [0, 1/8]
        This value is passed to the tike.ptycho.object.smoothness_constraint
        function.

    """

    def __init__(
        self,
        positivity_constraint=0,
        smoothness_constraint=0,
        use_adaptive_moment=False,
        vdecay=0.6,
        mdecay=0.666,
    ):
        self.positivity_constraint = positivity_constraint
        self.smoothness_constraint = smoothness_constraint
        self.use_adaptive_moment = use_adaptive_moment
        self.vdecay = vdecay
        self.mdecay = mdecay
        self.v = None
        self.m = None

    def put(self):
        """Copy to the current GPU memory."""
        if self.v is not None:
            self.v = cp.asarray(self.v)
        if self.m is not None:
            self.m = cp.asarray(self.m)
        return self

    def get(self):
        """Copy to the host CPU memory."""
        if self.v is not None:
            self.v = cp.asnumpy(self.v)
        if self.m is not None:
            self.m = cp.asnumpy(self.m)
        return self


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


def get_padded_object(scan, probe):
    """Return a ones-initialized object and shifted scan positions.

    An complex object array is initialized with shape such that the area
    covered by the probe is padded on each edge by a full probe width. The scan
    positions are shifted to be centered in this newly initialized object
    array.
    """
    # Shift scan positions to zeros
    scan[..., 0] -= np.min(scan[..., 0])
    scan[..., 1] -= np.min(scan[..., 1])

    # Add padding to scan positions of field-of-view / 8
    span = np.max(scan[..., 0]), np.max(scan[..., 1])
    scan[..., 0] += probe.shape[-2]
    scan[..., 1] += probe.shape[-1]

    height = 3 * probe.shape[-2] + int(span[0])
    width = 3 * probe.shape[-1] + int(span[1])

    return np.ones((height, width), dtype='complex64'), scan
