__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

import numpy as np

from .flow import _remap_lanczos
from .operator import Operator

class Pad(Operator):
    """Pad a stack of 2D images to the same shape but with unique pad_widths.
    """

    def fwd(self, unpadded, corner, padded_shape, **kwargs):
        assert np.all(np.asarray(unpadded.shape) <= padded_shape)
        assert self.xp.all(corner >= 0)
        # assert self.xp.all(corner + unpadded.shape[1:] <= padded_shape[1:])
        padded = self.xp.zeros(dtype=unpadded.dtype, shape=padded_shape)
        for i in range(padded.shape[0]):
            # yapf: disable
            padded[
                i,
                corner[i, 0]:corner[i, 0] + unpadded.shape[1],
                corner[i, 1]:corner[i, 1] + unpadded.shape[2]] = unpadded[i]
            # yapf: enable
        return padded

    def adj(self, padded, corner, unpadded_shape, **kwargs):
        assert np.all(np.asarray(unpadded_shape) <= padded.shape)
        assert self.xp.all(corner >= 0)
        # assert self.xp.all(corner + unpadded_shape[1:] <= padded.shape[1:])
        unpadded = self.xp.empty(dtype=padded.dtype, shape=unpadded_shape)
        for i in range(unpadded.shape[0]):
            # yapf: disable
            unpadded[i] = padded[
                i,
                corner[i, 0]:corner[i, 0] + unpadded.shape[1],
                corner[i, 1]:corner[i, 1] + unpadded.shape[2]]
            # yapf: enable
        return unpadded
