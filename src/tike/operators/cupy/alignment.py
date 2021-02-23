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
    """An alignment operator composed of pad, flow, and rotate operations.

    The operations are applied in the aforementioned order.

    Please see the help for the Pad, Flow, and Rotate operations for
    description of arguments.
    """

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
        shift,
        flow,
        padded_shape,
        angle,
        unpadded_shape=None,
        cval=0.0,
    ):
        unflowed = self.shift.fwd(
            a=self.pad.fwd(
                unpadded=unpadded,
                padded_shape=padded_shape,
                cval=cval,
            ),
            shift=shift,
            cval=cval,
        )
        if flow is None:
            return self.rotate.fwd(
                unrotated=unflowed,
                angle=angle,
                cval=cval,
            )
        else:
            if angle is not None:
                flow = flow + self.rotate._make_flow(unrotated=unflowed,
                                                     angle=angle)
            return self.flow.fwd(
                f=unflowed,
                flow=flow,
                cval=cval,
            )

    def adj(
        self,
        rotated,
        flow,
        shift,
        unpadded_shape,
        angle,
        padded_shape=None,
        cval=0.0,
    ):
        if flow is None:
            unflowed = self.rotate.adj(
                rotated=rotated,
                angle=angle,
                cval=cval,
            )
        else:
            if angle is not None:
                flow = flow + self.rotate._make_flow(unrotated=rotated,
                                                     angle=angle)
            unflowed = self.flow.adj(
                g=rotated,
                flow=flow,
                cval=cval,
            )
        return self.pad.adj(
            padded=self.shift.adj(
                a=unflowed,
                shift=shift,
                cval=cval,
            ),
            unpadded_shape=unpadded_shape,
            cval=cval,
        )

    def inv(
        self,
        rotated,
        flow,
        shift,
        unpadded_shape,
        angle,
        padded_shape=None,
        cval=0.0,
    ):
        return self.pad.adj(
            padded=self.shift.adj(
                a=self.flow.fwd(
                    f=self.rotate.fwd(
                        unrotated=rotated,
                        angle=angle if angle is None else -angle,
                        cval=cval,
                    ),
                    flow=flow if flow is None else -flow,
                    cval=cval,
                ),
                shift=shift,
                cval=cval,
            ),
            unpadded_shape=unpadded_shape,
            cval=cval,
        )
