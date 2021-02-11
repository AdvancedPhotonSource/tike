__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

import numpy as np

from .flow import _remap_lanczos
from .operator import Operator


class Pad(Operator):
    """Pad a stack of 2D images to the same shape but with unique pad_widths.

    By default, no padding is applied and/or the padding is applied
    symmetrically.
    """

    def fwd(self, unpadded, corner=None, padded_shape=None, cval=0.0, **kwargs):
        """Pad the unpadded images with cval.

        Parameters
        ----------
        corner : (N, 2)
            The min corner of the images in the padded array.
        padded_shape : 3-tuple
            The desired shape after padding. First element should be N.
        unpadded_shape : 3-tuple
            See padded_shape.
        cval : complex64
            The value to use for padding.
        """
        if padded_shape is None:
            padded_shape = unpadded.shape
        if corner is None:
            corner = self.xp.tile(
                (((padded_shape[-2] - unpadded.shape[-2]) // 2,
                  (padded_shape[-1] - unpadded.shape[-1]) // 2)),
                (padded_shape[0], 1),
            )

        padded = self.xp.empty(shape=padded_shape, dtype=unpadded.dtype)
        padded[:] = cval
        for i in range(padded.shape[0]):
            lo0, hi0 = corner[i, 0], corner[i, 0] + unpadded.shape[-2]
            lo1, hi1 = corner[i, 1], corner[i, 1] + unpadded.shape[-1]
            assert lo0 >= 0 and lo1 >= 0
            assert hi0 <= padded.shape[-2] and hi1 <= padded.shape[-1]
            padded[i][lo0:hi0, lo1:hi1] = unpadded[i]
        return padded

    def adj(self, padded, corner=None, unpadded_shape=None, **kwargs):
        """Strip the edges from the padded images.

        Parameters
        ----------
        corner : (N, 2)
            The min corner of the images in the padded array.
        padded_shape : 3-tuple
            The desired shape after padding. First element should be N.
        unpadded_shape : 3-tuple
            See padded_shape.
        cval : complex64
            The value to use for padding.
        """
        if unpadded_shape is None:
            unpadded_shape = padded.shape
        if corner is None:
            corner = self.xp.tile(
                (((padded.shape[-2] - unpadded_shape[-2]) // 2,
                  (padded.shape[-1] - unpadded_shape[-1]) // 2)),
                (padded.shape[0], 1),
            )

        unpadded = self.xp.empty(shape=unpadded_shape, dtype=padded.dtype)
        for i in range(padded.shape[0]):
            lo0, hi0 = corner[i, 0], corner[i, 0] + unpadded.shape[-2]
            lo1, hi1 = corner[i, 1], corner[i, 1] + unpadded.shape[-1]
            assert lo0 >= 0 and lo1 >= 0
            assert hi0 <= padded.shape[-2] and hi1 <= padded.shape[-1]
            unpadded[i] = padded[i][lo0:hi0, lo1:hi1]
        return unpadded

    inv = adj
