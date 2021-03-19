__author__ = "Viktor Nikitin, Daniel Ching"
__copyright__ = "Copyright (c) 2021, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

from .operator import Operator


class Gradient(Operator):
    """Returns the Gradient approximation of a 3D array."""

    def fwd(self, u):
        """Forward operator for regularization."""
        res = self.xp.empty((3, *u.shape), dtype=u.dtype)
        res[0, :, :, :-1] = u[:, :, 1:] - u[:, :, :-1]
        res[0, :, :,  -1] = u[:, :,  0] - u[:, :,  -1]
        res[1, :, :-1, :] = u[:, 1:, :] - u[:, :-1, :]
        res[1, :,  -1, :] = u[:,  0, :] - u[:,  -1, :]
        res[2, :-1, :, :] = u[1:, :, :] - u[:-1, :, :]
        res[2,  -1, :, :] = u[ 0, :, :] - u[ -1, :, :]
        res *= 1 / self.xp.sqrt(3)  # normalization
        return res

    def adj(self, g):
        """Adjoint operator for regularization."""
        res = self.xp.empty(g.shape[1:], g.dtype)
        res[:, :, 1:] = g[0, :, :, 1:] - g[0, :, :, :-1]
        res[:, :, 0] = g[0, :, :, 0] - g[0, :, :, -1]
        res[:, 1:, :] += g[1, :, 1:, :] - g[1, :, :-1, :]
        res[:, 0, :] += g[1, :, 0, :] - g[1, :, -1, :]
        res[1:, :, :] += g[2, 1:, :, :] - g[2, :-1, :, :]
        res[0, :, :] += g[2, 0, :, :] - g[2, -1, :, :]
        res *= -1 / self.xp.sqrt(3)  # normalization
        return res
