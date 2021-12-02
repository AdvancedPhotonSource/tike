__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2021, UChicago Argonne, LLC."

try:
    from importlib.resources import files
except ImportError:
    # Backport for python<3.9 available as importlib_resources package
    from importlib_resources import files

import cupy as cp
import numpy as np

from .operator import Operator

_cu_source = files('tike.operators.cupy').joinpath('convolution.cu').read_text()
_fwd_patch = cp.RawKernel(_cu_source, "fwd_patch")
_adj_patch = cp.RawKernel(_cu_source, "adj_patch")


def _next_power_two(v):
    """Return the next highest power of 2 of 32-bit v.

    https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
    """
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    return v + 1


class Patch(Operator):
    """Extract (zero-padded) patches from images at provided positions.

    Parameters
    ----------
    images : (..., H, W) complex64
        The complex wavefront modulation of the object.
    positions : (..., N, 2) float32
        Coordinates of the minimum corner of the patches in the image grid.
    patches : (..., N * nrepeat, width+, width+) complex64
        The extracted (zero-padded) patches.
    patch_width : int
        The width of the unpadded patches.
    """

    def fwd(
        self,
        images,
        positions,
        patches=None,
        patch_width=None,
        height=None,
        width=None,
        nrepeat=1,
    ):
        patch_width = patches.shape[-1] if patch_width is None else patch_width
        if patches is None:
            patches = cp.zeros(
                (*positions.shape[:-2], positions.shape[-2] * nrepeat,
                 patch_width, patch_width),
                dtype='complex64',
            )
        assert patch_width <= patches.shape[-1]
        assert images.shape[:-2] == positions.shape[:-2]
        assert positions.shape[:-2] == patches.shape[:-3], (positions.shape,
                                                            patches.shape)
        assert positions.shape[-2] * nrepeat == patches.shape[-3]
        assert positions.shape[-1] == 2, positions.shape
        assert images.dtype == 'complex64'
        assert patches.dtype == 'complex64'
        assert positions.dtype == 'float32'
        nimage = int(np.prod(images.shape[:-2]))
        grids = (
            positions.shape[-2],
            nimage,
            patch_width,
        )
        blocks = (min(_next_power_two(patch_width),
                      _fwd_patch.attributes['max_threads_per_block']),)
        _fwd_patch(
            grids,
            blocks,
            (
                images,
                patches,
                positions,
                nimage,
                *images.shape[-2:],
                positions.shape[-2],
                nrepeat,
                patch_width,
                patches.shape[-1],
            ),
        )
        return patches

    def adj(
        self,
        positions,
        patches,
        images=None,
        patch_width=None,
        height=None,
        width=None,
        nrepeat=1,
    ):
        patch_width = patches.shape[-1] if patch_width is None else patch_width
        assert patch_width <= patches.shape[-1]
        if images is None:
            images = cp.zeros(
                (*positions.shape[:-2], height, width),
                dtype='complex64',
            )
        assert images.shape[:-2] == positions.shape[:-2]
        assert positions.shape[:-2] == patches.shape[:-3], (positions.shape,
                                                            patches.shape)
        assert positions.shape[-2] * nrepeat == patches.shape[-3], (
            positions.shape, nrepeat, patches.shape)
        assert positions.shape[-1] == 2
        assert images.dtype == 'complex64'
        assert patches.dtype == 'complex64'
        assert positions.dtype == 'float32'
        nimage = int(np.prod(images.shape[:-2]))
        grids = (
            positions.shape[-2],
            nimage,
            patch_width,
        )
        blocks = (min(_next_power_two(patch_width),
                      _adj_patch.attributes['max_threads_per_block']),)
        _adj_patch(
            grids,
            blocks,
            (
                images,
                patches,
                positions,
                nimage,
                *images.shape[-2:],
                positions.shape[-2],
                nrepeat,
                patch_width,
                patches.shape[-1],
            ),
        )
        return images
