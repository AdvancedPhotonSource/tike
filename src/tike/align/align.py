__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    "reconstruct",
    "simulate",
]

import logging
import numpy as np

from tike.operators import Shift
from tike.align.solvers import cross_correlation

logger = logging.getLogger(__name__)


def simulate(
        unaligned,
        shift,
        **kwargs
):  # yapf: disable
    """Return unaligned shifted by shift."""
    assert unaligned.ndim > 2
    assert shift.ndim > 1
    with Shift() as operator:
        data = operator.fwd(
            a=operator.asarray(unaligned, dtype='complex64'),
            shift=operator.asarray(shift, dtype='float32'),
        )
        assert data.dtype == 'complex64', data.dtype
        return operator.asnumpy(data)


def reconstruct(
        data,
        unaligned,
        shift=None,
        num_iter=1, rtol=-1, **kwargs
):  # yapf: disable
    """Solve the alignment problem.

    Parameters
    ----------
    unaligned : (..., H, W) complex64
        The images to be aligned with data.
    shift : (..., 2) float32
        The inital guesses for the shifts.
    rtol : float
        Terminate early if the relative decrease of the cost function is
        less than this amount.

    """
    if shift is None:
        shift = np.zeros((*unaligned.shape[:-2], 2), dtype='float32')
    # Initialize an operator.
    with Shift() as operator:
        # send any array-likes to device
        data = operator.asarray(data, dtype='complex64')
        unaligned = operator.asarray(unaligned, dtype='complex64')
        result = {
            'shift': operator.asarray(shift, dtype='float32'),
        }
        for key, value in kwargs.items():
            if np.ndim(value) > 0:
                kwargs[key] = operator.asarray(value)

        logger.info("{} on {:,d} - {:,d} by {:,d} images for {:,d} "
                    "iterations.".format('cross_correlation', *data.shape,
                                         num_iter))

        kwargs.update(result)
        result = cross_correlation(
            operator,
            data=data,
            unaligned=unaligned,
            **kwargs,
        )

    return {k: operator.asnumpy(v) for k, v in result.items()}
