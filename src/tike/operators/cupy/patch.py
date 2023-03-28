__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2021, UChicago Argonne, LLC."

try:
    from importlib.resources import files
except ImportError:
    # Backport for python<3.9 available as importlib_resources package
    from importlib_resources import files

import cupy as cp
import numpy.typing as npt
import numpy as np

from .operator import Operator

kernels = [
    'fwd_patch<float2,float2,float>',
    'adj_patch<float2,float2,float>',
    'fwd_patch<double2,double2,float>',
    'adj_patch<double2,double2,float>',
    'fwd_patch<float2,double2,float>',
    'adj_patch<float2,double2,float>',
    'fwd_patch<double2,float2,float>',
    'adj_patch<double2,float2,float>',
    'fwd_patch<float2,float2,double>',
    'adj_patch<float2,float2,double>',
    'fwd_patch<double2,double2,double>',
    'adj_patch<double2,double2,double>',
    'fwd_patch<float2,double2,double>',
    'adj_patch<float2,double2,double>',
    'fwd_patch<double2,float2,double>',
    'adj_patch<double2,float2,double>',
]

_patch_module = cp.RawModule(
    code=files('tike.operators.cupy').joinpath('convolution.cu').read_text(),
    name_expressions=kernels,
    options=('--std=c++11',),
)

typename = {
    np.dtype('complex64'): 'float2',
    np.dtype('float32'): 'float',
    np.dtype('complex128'): 'double2',
    np.dtype('float64'): 'double',
}


def _next_power_two(v: int) -> int:
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
           OR (..., L, width+, width+) complex64
        The extracted (zero-padded) patches. (N * nrepeat) = K * L, K >= nrepeat
    patch_width : int
        The width of the unpadded patches.
    """

    def fwd(
        self,
        images: npt.NDArray[np.csingle],
        positions: npt.NDArray[np.single],
        patches: npt.NDArray[np.csingle] = None,
        patch_width: int = 0,
        height: int = 0,
        width: int = 0,
        nrepeat: int = 1,
    ):
        patch_width = patches.shape[-1] if patch_width == 0 else patch_width
        if patches is None:
            patches = cp.zeros_like(
                images,
                shape=(*positions.shape[:-2], positions.shape[-2] * nrepeat,
                       patch_width, patch_width),
            )
        assert patch_width <= patches.shape[-1]
        assert images.shape[:-2] == positions.shape[:-2]
        assert positions.shape[:-2] == patches.shape[:-3], (positions.shape,
                                                            patches.shape)
        assert positions.shape[-2] * nrepeat == patches.shape[-3]
        assert positions.shape[-1] == 2, positions.shape
        nimage = int(np.prod(images.shape[:-2]))

        _fwd_patch = _patch_module.get_function(
            f'fwd_patch<{typename[patches.dtype]},{typename[images.dtype]},{typename[positions.dtype]}>'
        )

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
        positions: npt.NDArray[np.single],
        patches: npt.NDArray[np.csingle],
        images: npt.NDArray[np.csingle] = None,
        patch_width: int = 0,
        height: int = 0,
        width: int = 0,
        nrepeat: int = 1,
    ):
        patch_width = patches.shape[-1] if patch_width == 0 else patch_width
        assert patch_width <= patches.shape[-1]
        if images is None:
            images = cp.zeros_like(
                patches,
                shape=(*positions.shape[:-2], height, width),
            )
        leading = images.shape[:-2]
        height, width = images.shape[-2:]
        assert positions.shape[:-2] == leading
        N = positions.shape[-2]
        assert positions.shape[-1] == 2
        assert patches.shape[:-3] == leading
        K = patches.shape[-3]
        assert (N * nrepeat) % K == 0 and K >= nrepeat
        assert patches.shape[-1] == patches.shape[-2]
        nimage = int(np.prod(images.shape[:-2]))

        _adj_patch = _patch_module.get_function(
            f'adj_patch<{typename[patches.dtype]},{typename[images.dtype]},{typename[positions.dtype]}>'
        )

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
                height,
                width,
                N,
                nrepeat,
                patch_width,
                patches.shape[-1],
                K,
            ),
        )
        return images
