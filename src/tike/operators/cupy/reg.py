__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

import cupy as cp
from tike.operators import numpy
from .operator import Operator


class Reg(Operator, numpy.Reg):
    pass