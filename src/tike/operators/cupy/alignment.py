"""Defines an alignment operator."""

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

import numpy as np

from .operator import Operator
from .rotate import Rotate
from .pad import Pad


class Alignment(Operator):
    """An alignment operator composed of pad and rotate operations."""

    def __init__(self):
        """Please see help(Alignment) for more info."""
        self.pad = Pad()
        self.rotate = Rotate()

    def __enter__(self):
        self.pad.__enter__()
        self.rotate.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self.pad.__exit__(type, value, traceback)
        self.rotate.__exit__(type, value, traceback)

    def fwd(self, unpadded, corner, padded_shape, angle, **kwargs):
        return self.rotate.fwd(
            unrotated=self.pad.fwd(
                unpadded=unpadded,
                corner=corner,
                padded_shape=padded_shape,
            ),
            angle=angle,
        )

    def adj(self, rotated, corner, unpadded_shape, angle, **kwargs):
        return self.pad.adj(
            padded=self.rotate.adj(
                rotated=rotated,
                angle=angle,
            ),
            corner=corner,
            unpadded_shape=unpadded_shape,
        )
