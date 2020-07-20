__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

from tike.operators import numpy
from .operator import Operator


class Flow(Operator, numpy.Flow):
    pass
