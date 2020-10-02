__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

import numpy as np

from .flow import _remap_lanczos
from .operator import Operator


class Rotate(Operator):
    """Rotate a stack of 2D images along last two dimensions."""

    def _make_grid(self, unrotated, angle):
        """Return the points on the rotated grid."""
        cos, sin = np.cos(angle), np.sin(angle)
        shifti = (unrotated.shape[-2] - 1) / 2.0
        shiftj = (unrotated.shape[-1] - 1) / 2.0

        i, j = self.xp.mgrid[0:unrotated.shape[-2],
                             0:unrotated.shape[-1]].astype('float32')

        i -= shifti
        j -= shiftj

        i1 = (+cos * i + sin * j) + shifti
        j1 = (-sin * i + cos * j) + shiftj

        return self.xp.stack([i1.ravel(), j1.ravel()], axis=-1)

    def fwd(self, unrotated, angle):
        if angle is None:
            return unrotated
        f = unrotated
        g = self.xp.zeros_like(f)

        # Compute rotated coordinates
        coords = self._make_grid(f, angle)

        # Reshape into stack of 2D images
        shape = f.shape
        h, w = shape[-2:]
        f = f.reshape(-1, h, w)
        g = g.reshape(-1, h * w)

        for i in range(len(f)):
            _remap_lanczos(f[i], coords, 2, g[i], fwd=True)

        return g.reshape(shape)

    def adj(self, rotated, angle):
        if angle is None:
            return rotated
        g = rotated
        f = self.xp.zeros_like(g)

        # Compute rotated coordinates
        coords = self._make_grid(f, angle)

        # Reshape into stack of 2D images
        shape = f.shape
        h, w = shape[-2:]
        f = f.reshape(-1, h, w)
        g = g.reshape(-1, h * w)

        for i in range(len(f)):
            _remap_lanczos(f[i], coords, 2, g[i], fwd=False)

        return f.reshape(shape)
