from tike.operators import numpy
from .operator import Operator


class Lamino(Operator, numpy.Lamino):
    def __init__(self, *args, **kwargs):
        super(Lamino, self).__init__(
            *args,
            **kwargs,
        )
