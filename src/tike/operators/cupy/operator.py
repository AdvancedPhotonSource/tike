from abc import ABC

import cupy


class Operator(ABC):

    xp = cupy

    @classmethod
    def asarray(cls, *args, **kwargs):
        return cupy.asarray(*args, **kwargs)

    @classmethod
    def asnumpy(cls, *args, **kwargs):
        return cupy.asnumpy(*args, **kwargs)
