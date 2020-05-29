from importlib_resources import files

import cupy as cp

from tike.operators import numpy
from .operator import Operator

_cu_source = files('tike.operators.cupy').joinpath('convolution.cu').read_text()
_patch_kernel = cp.RawKernel(_cu_source, "patch")


class Convolution(Operator, numpy.Convolution):

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def _patch(self, patches, psi, scan, fwd=True):
        max_thread = min(self.probe_shape**2,
                         _patch_kernel.attributes['max_threads_per_block'])
        blocks = (max_thread,)
        grids = (
            -(-self.probe_shape**2 // max_thread),  # ceil division
            scan.shape[-2],
            self.ntheta,
        )
        _patch_kernel(
            grids,
            blocks,
            (psi, patches, scan, self.ntheta, self.nz, self.n, scan.shape[-2],
             self.probe_shape, patches.shape[-1], fwd),
        )
        if fwd:
            return patches
        else:
            return psi
