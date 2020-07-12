__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

import numpy as np
from numpy.fft import fft2, ifft2

from .operator import Operator


class Reg(Operator):
    """3D Gradient and divergence operators for regularization."""

    def fwd(self, u):
        """3D gradient operator"""
        res = self.xp.zeros([3, *u.shape], dtype='float32')
        res[0, :, :, :-1] = u[:, :, 1:]-u[:, :, :-1]
        res[1, :, :-1, :] = u[:, 1:, :]-u[:, :-1, :]
        res[2, :-1, :, :] = u[1:, :, :]-u[:-1, :, :]
        return res

    def adj(self, gr):
        """3D negative divergence operator"""
        res = self.xp.zeros(gr.shape[1:], dtype='float32')
        res[:, :, 1:] = gr[0, :, :, 1:]-gr[0, :, :, :-1]
        res[:, :, 0] = gr[0, :, :, 0]
        res[:, 1:, :] += gr[1, :, 1:, :]-gr[1, :, :-1, :]
        res[:, 0, :] += gr[1, :, 0, :]
        res[1:, :, :] += gr[2, 1:, :, :]-gr[2, :-1, :, :]
        res[0, :, :] += gr[2, 0, :, :]
        return -res
        
    
    