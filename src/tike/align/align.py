__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    "reconstruct",
    "simulate",
    "invert",
]

import logging
import numpy as np

from tike.operators import Alignment
from tike.align import solvers

logger = logging.getLogger(__name__)


def simulate(
        original,
        **kwargs
):  # yapf: disable
    """Return original shifted by shift."""
    with Alignment() as operator:
        for key, value in kwargs.items():
            if not isinstance(value, tuple) and np.ndim(value) > 0:
                kwargs[key] = operator.asarray(value)
        unaligned = operator.fwd(
            operator.asarray(original, dtype='complex64'),
            **kwargs,
        )
        assert unaligned.dtype == 'complex64', unaligned.dtype
        return operator.asnumpy(unaligned)


def invert(
        original,
        **kwargs
):  # yapf: disable
    """Return original shifted by shift."""
    with Alignment() as operator:
        for key, value in kwargs.items():
            if not isinstance(value, tuple) and np.ndim(value) > 0:
                kwargs[key] = operator.asarray(value)
        unaligned = operator.inv(
            operator.asarray(original, dtype='complex64'),
            **kwargs,
        )
        assert unaligned.dtype == 'complex64', unaligned.dtype
        return operator.asnumpy(unaligned)


def reconstruct(
        original,
        unaligned,
        algorithm,
        num_iter=1, rtol=-1, **kwargs
):  # yapf: disable
    """Solve the alignment problem; returning either the original or the shift.

    Parameters
    ----------
    unaligned, original: (..., H, W) complex64
        The images to be aligned.
    rtol : float
        Terminate early if the relative decrease of the cost function is
        less than this amount.

    """
    if algorithm in solvers.__all__:
        with Alignment() as operator:
            for key, value in kwargs.items():
                if not isinstance(value, tuple) and np.ndim(value) > 0:
                    kwargs[key] = operator.asarray(value)
            logger.info("{} on {:,d} - {:,d} by {:,d} images for {:,d} "
                        "iterations.".format(algorithm, *unaligned.shape,
                                             num_iter))
            result = getattr(solvers, algorithm)(
                operator,
                original=operator.asarray(original, dtype='complex64'),
                unaligned=operator.asarray(unaligned, dtype='complex64'),
                num_iter=num_iter,
                **kwargs,
            )
        return {k: operator.asnumpy(v) for k, v in result.items()}
    else:
        raise ValueError(
            "The '{}' algorithm is not an available.".format(algorithm))
