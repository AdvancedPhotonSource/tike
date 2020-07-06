__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

import numpy as np
from scipy.ndimage import map_coordinates

from .operator import Operator


class Flow(Operator):
    """Map input 2D array to new coordinates by interpolation.

    This operator is based on scipy's map_coordinates and peforms a non-affine
    deformation of a series of 2D images.
    """

    def fwd(self, f, flow):
        """Apply arbitary shifts to individuals pixels of f.

        Parameters
        ----------
        f (..., H, W) complex64
            A stack of arrays to be deformed.
        flow (..., H, W, 2) float32
            The displacements to be applied to each pixel along the last two
            dimensions.

        """
        # Convert from displacements to coordinates
        h, w = flow.shape[-3:-1]
        coords = -flow.copy()
        coords[..., 0] += self.xp.arange(h)[:, None]
        coords[..., 1] += self.xp.arange(w)

        coords = coords.reshape(-1, h, w, 2)
        shape = f.shape
        f = f.reshape(-1, h, w)
        g = self.xp.empty_like(f)

        for i in range(len(f)):
            # Move flow dimension to front for map_coordinates API
            g.real[i] = map_coordinates(
                input=f.real[i],
                coordinates=self.xp.moveaxis(coords[i], -1, 0),
                output=g.real[i],
            )
            g.imag[i] = map_coordinates(
                input=f.imag[i],
                coordinates=self.xp.moveaxis(coords[i], -1, 0),
                output=g.imag[i],
            )

        return g.reshape(shape)
