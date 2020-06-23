import cupy as cp
from cupyx.scipy.fft import fftn, ifftn
from cupyx.scipy.fftpack import get_fft_plan

from tike.operators import numpy
from .cache import CachedFFT
from .operator import Operator


class Propagation(Operator, CachedFFT, numpy.Propagation):
    """A Fourier-based free-space propagation using CuPy."""
