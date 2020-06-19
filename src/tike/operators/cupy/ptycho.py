from tike.operators import numpy
from .convolution import Convolution
from .propagation import Propagation
from .operator import Operator

import cupy as cp


class Ptycho(Operator, numpy.Ptycho):

    def __init__(self, *args, **kwargs):
        super(Ptycho, self).__init__(
            *args,
            propagation=Propagation,
            diffraction=Convolution,
            **kwargs,
        )

    # Multi-GPU related

    # def grad_device(self, gpu_id, data, psi, scan, probe):
    #     with cp.cuda.Device(gpu_id):
    #         return self.grad(data, psi, scan, probe)

    # def cost_device(self, gpu_id, data, psi, scan, probe, n=-1, mode=None):
    #     with cp.cuda.Device(gpu_id):
    #         return self.cost(data, psi, scan, probe)

    # def update_device(self, gpu_id, psi, gamma, dir):
    #     with cp.cuda.Device(gpu_id):
    #         return psi + gamma * dir
