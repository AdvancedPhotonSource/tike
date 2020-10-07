"""Defines an alignment operator."""

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

import numpy as np

from .flow import Flow
from .operator import Operator
from .pad import Pad
from .rotate import Rotate
from .shift import Shift


class Alignment(Operator):
    """An alignment operator composed of pad and rotate operations."""

    def __init__(self):
        """Please see help(Alignment) for more info."""
        self.flow = Flow()
        self.pad = Pad()
        self.rotate = Rotate()
        self.shift = Shift()

    def __enter__(self):
        self.flow.__enter__()
        self.pad.__enter__()
        self.rotate.__enter__()
        self.shift.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self.flow.__exit__(type, value, traceback)
        self.pad.__exit__(type, value, traceback)
        self.rotate.__exit__(type, value, traceback)
        self.shift.__exit__(type, value, traceback)

    def fwd(
        self,
        unpadded,
        flow,
        padded_shape,
        angle,
        unpadded_shape=None,
        cval=0.0,
    ):
        return self.rotate.fwd(
            unrotated=self.flow.fwd(
                f=self.pad.fwd(
                    unpadded=unpadded,
                    padded_shape=padded_shape,
                    cval=cval,
                ),
                flow=flow,
                cval=cval,
            ),
            angle=angle,
            cval=cval,
        )

    def adj(
        self,
        rotated,
        flow,
        unpadded_shape,
        angle,
        padded_shape=None,
        cval=0.0,
    ):
        return self.pad.adj(
            padded=self.flow.adj(
                g=self.rotate.adj(
                    rotated=rotated,
                    angle=angle,
                    cval=cval,
                ),
                flow=flow,
                cval=cval,
            ),
            unpadded_shape=unpadded_shape,
            cval=cval,
        )
