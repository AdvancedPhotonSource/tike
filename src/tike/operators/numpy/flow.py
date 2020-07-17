__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

import numpy as np
from scipy.ndimage import map_coordinates

from .operator import Operator


def _lanzcos(xp, x, a):
    return xp.sinc(x) * xp.sinc(x / a)


def _remap_lanzcos(xp, Fe, x, m, F=None):
    """Lanzcos resampling from grid Fe to points x.

    At the edges, the Lanzcos filter wraps around.

    Parameters
    ----------
    xp : module
        The array module for this implementation
    Fe : (H, W)
        The function at equally spaced samples.
    x : (N, 2) float32
        The non-uniform sample positions on the grid.
    m : int > 0
        The lanzcos filter is 2m + 1 wide.

    Returns
    -------
    F : (N, )
        The values at the non-uniform samples.
    """
    # NOTE: This irregular convolution is very similar to the gather function
    # from usfft
    assert Fe.ndim == 2
    assert x.ndim == 2 and x.shape[-1] == 2
    assert m > 0
    F = xp.zeros(x.shape[:-1], dtype=Fe.dtype) if F is None else F
    assert F.shape == x.shape[:-1], F.dtype == Fe.dtype
    n = Fe.shape[-2:]
    # ell is the integer center of the kernel
    ell = xp.floor(x).astype('int32')
    for i0 in range(-m, m + 1):
        kern0 = _lanzcos(xp, ell[..., 0] + i0 - x[..., 0], m)
        for i1 in range(-m, m + 1):
            kern1 = _lanzcos(xp, ell[..., 1] + i1 - x[..., 1], m)
            # Indexing Fe here causes problems for a stack of images
            F += Fe[(ell[..., 0] + i0) % n[0],
                    (ell[..., 1] + i1) % n[1]] * kern0 * kern1
    return F


class Flow(Operator):
    """Map input 2D array to new coordinates by interpolation.

    This operator is based on scipy's map_coordinates and peforms a non-affine
    deformation of a series of 2D images.
    """

    @classmethod
    def _map_coordinates(cls, *args, **kwargs):
        return map_coordinates(*args, **kwargs)

    def fwd(self, f, flow, filter_size=5):
        """Remap individual pixels of f with Lanzcos filtering.

        Parameters
        ----------
        f (..., H, W) complex64
            A stack of arrays to be deformed.
        flow (..., H, W, 2) float32
            The displacements to be applied to each pixel along the last two
            dimensions.
        filter_size : int
            The width of the Lanzcos filter. Automatically rounded up to an
            odd positive integer.
        """
        # Convert from displacements to coordinates
        h, w = flow.shape[-3:-1]
        coords = -flow.copy()
        coords[..., 0] += self.xp.arange(h)[:, None]
        coords[..., 1] += self.xp.arange(w)

        # Reshape into stack of 2D images
        shape = f.shape
        coords = coords.reshape(-1, h * w, 2)
        f = f.reshape(-1, h, w)
        g = self.xp.zeros_like(f).reshape(-1, h * w)

        a = max(0, (filter_size) // 2)
        for i in range(len(f)):
            _remap_lanzcos(self.xp, f[i], coords[i], a, g[i])

        return g.reshape(shape)
