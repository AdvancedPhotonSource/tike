import cupy as cp
from cupyx.scipy.fft import fftn, ifftn
from cupyx.scipy.fftpack import get_fft_plan

from tike.operators import numpy
from .operator import Operator


class Propagation(Operator, numpy.Propagation):
    """A Fourier-based free-space propagation using CuPy."""

    def __enter__(self):
        # TODO: initialize fftplan cache
        return self

    def __exit__(self, type, value, traceback):
        # TODO: delete fftplan cache
        pass

    def _fft2(self, *args, overwrite=False, **kwargs):
        return fftn(*args, overwrite_x=overwrite, **kwargs)

    def _ifft2(self, *args, overwrite=False, **kwargs):
        return ifftn(*args, overwrite_x=overwrite, **kwargs)
