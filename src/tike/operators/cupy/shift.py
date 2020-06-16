__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

import cupy as cp
from cupyx.scipy.fft import fftn, ifftn
from cupyx.scipy.fftpack import get_fft_plan

from tike.operators import numpy
from .cache import CachedFFT
from .operator import Operator


class Shift(Operator, CachedFFT, numpy.Shift):
    pass
