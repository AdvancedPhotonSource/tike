from tike.operators import numpy
from .convolution import Convolution
from .propagation import Propagation
from .operator import Operator


class Ptycho(Operator, numpy.Ptycho):

    def __init__(self, *args, **kwargs):
        super(Ptycho, self).__init__(
            *args,
            propagation=Propagation,
            diffraction=Convolution,
            **kwargs,
        )
