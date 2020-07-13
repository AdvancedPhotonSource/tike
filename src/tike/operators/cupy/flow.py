__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

from cupyx.scipy.ndimage import map_coordinates

from tike.operators import numpy
from .operator import Operator

class Flow(Operator, numpy.Flow):

    @classmethod
    def _map_coordinates(cls, *args, **kwargs):
        # https://github.com/cupy/cupy/pull/2813
        # Will not work until CuPy>=8
        return map_coordinates(*args, **kwargs)
