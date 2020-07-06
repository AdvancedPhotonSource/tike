__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    "reconstruct",
    "simulate",
]

import logging
import numpy as np

from tike.operators import Shift, Flow
from tike.align import solvers

logger = logging.getLogger(__name__)


def simulate(
        unaligned,
        shift,
        **kwargs
):  # yapf: disable
    """Return unaligned shifted by shift."""
    assert unaligned.ndim > 2
    if shift.shape == (*unaligned.shape, 2):
        Operator = Flow
    elif shift.shape == (*unaligned.shape[:-2], 2):
        Operator = Shift
    else:
        raise ValueError(
            'There must be one shift per image or one shift per pixel.')

    with Operator() as operator:
        data = operator.fwd(
            operator.asarray(unaligned, dtype='complex64'),
            operator.asarray(shift, dtype='float32'),
        )
        assert data.dtype == 'complex64', data.dtype
        return operator.asnumpy(data)


def reconstruct(
        data,
        unaligned,
        algorithm,
        shift=None,
        num_iter=1, rtol=-1, **kwargs
):  # yapf: disable
    """Solve the alignment problem.

    Parameters
    ----------
    unaligned : (..., H, W) complex64
        The images to be aligned with data.
    shift : (..., 2), (..., H, W, 2) float32
        The inital guesses for the shifts.
    rtol : float
        Terminate early if the relative decrease of the cost function is
        less than this amount.

    """
    if algorithm in solvers.__all__:

        Operator = Flow if algorithm == 'farneback' else Shift

        # Initialize an operator.
        with Operator() as operator:
            # send any array-likes to device
            data = operator.asarray(data, dtype='complex64')
            unaligned = operator.asarray(unaligned, dtype='complex64')
            result = {}
            for key, value in kwargs.items():
                if np.ndim(value) > 0:
                    kwargs[key] = operator.asarray(value)

            logger.info("{} on {:,d} - {:,d} by {:,d} images for {:,d} "
                        "iterations.".format(algorithm, *data.shape,
                                                num_iter))

            kwargs.update(result)
            result = getattr(solvers, algorithm)(
                operator,
                data=data,
                unaligned=unaligned,
                **kwargs,
            )

        return {k: operator.asnumpy(v) for k, v in result.items()}
    else:
        raise ValueError(
            "The '{}' algorithm is not an available.".format(algorithm))
