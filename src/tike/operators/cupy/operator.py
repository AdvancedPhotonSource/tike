from abc import ABC

import cupy


class Operator(ABC):

    xp = cupy

    @classmethod
    def asarray(cls, *args, device=None, **kwargs):
        with cupy.cuda.Device(device):
            return cupy.asarray(*args, **kwargs)

    @classmethod
    def asnumpy(cls, *args, **kwargs):
        return cupy.asnumpy(*args, **kwargs)
