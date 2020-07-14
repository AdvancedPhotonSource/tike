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
        original,
        shift,
        **kwargs
):  # yapf: disable
    """Return original shifted by shift."""
    assert original.ndim > 2
    if shift.shape == (*original.shape, 2):
        Operator = Flow
    elif shift.shape == (*original.shape[:-2], 2):
        Operator = Shift
    else:
        raise ValueError(
            'There must be one shift per image or one shift per pixel.')

    with Operator() as operator:
        unaligned = operator.fwd(
            operator.asarray(original, dtype='complex64'),
            operator.asarray(shift, dtype='float32'),
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
    shift : (..., 2), (..., H, W, 2) float32
        The displacements of pixels from original to unaligned.
    rtol : float
        Terminate early if the relative decrease of the cost function is
        less than this amount.

    """
    if algorithm in solvers.__all__:

        # Initialize an operator.
        with Flow() as operator:
            # send any array-likes to device
            unaligned = operator.asarray(unaligned, dtype='complex64')
            original = operator.asarray(original, dtype='complex64')
            result = {}
            for key, value in kwargs.items():
                if np.ndim(value) > 0:
                    kwargs[key] = operator.asarray(value)

            logger.info("{} on {:,d} - {:,d} by {:,d} images for {:,d} "
                        "iterations.".format(algorithm, *unaligned.shape,
                                             num_iter))

            kwargs.update(result)
            result = getattr(solvers, algorithm)(
                operator,
                original=original,
                unaligned=unaligned,
                num_iter=num_iter,
                **kwargs,
            )

        return {k: operator.asnumpy(v) for k, v in result.items()}
    else:
        raise ValueError(
            "The '{}' algorithm is not an available.".format(algorithm))
