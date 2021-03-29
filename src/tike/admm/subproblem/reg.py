import logging

import dxchange
import numpy as np

import tike.regularization
from . import update_penalty

logger = logging.getLogger(__name__)


def reg(
    # constants
    comm,
    u,
    # updated
    omega,
    dual,
    penalty,
    Ju0=None,
    # parameters
    folder=None,
    save_result=False,
):
    """Update omega, the regularized object."""
    op = tike.operators.Gradient()

    omega = tike.regularization.soft_threshold(
        op,
        x=u
        dual=dual,
        penalty=penalty,
        alpha=alpha,
    )

    logger.info('Update regularization lambdas and rhos.')

    dual += penalty * (omega - Ju)

    if Ju0 is not None:
        penalty = update_penalty(comm, omega, Ju, Ju0, penalty)

    Ju0 = Ju

    return (
        omega,
        dual,
        penalty,
        Ju0,
    )
