import logging

import numpy as np

from tike.opt import conjugate_gradient, line_search

logger = logging.getLogger(__name__)

def combined(
    op,
    data, probe, scan, psi,
    recover_psi=True, recover_probe=True,
    cg_iter=4,
    **kwargs
):  # yapf: disable
    """Solve the ptychography problem using a combined approach.

    """
    if recover_psi:
        psi, cost = update_object(op, data, psi, scan, probe, num_iter=cg_iter)

    if recover_probe:
        probe, cost = update_probe(op, data, psi, scan, probe, num_iter=cg_iter)

    return {'psi': psi, 'probe': probe, 'cost': cost, 'scan': scan}


def update_probe(op, data, psi, scan, probe, num_iter=1):
    """Solve the probe recovery problem."""

    def cost_function(probe):
        return op.cost(data, psi, scan, probe)

    def grad(probe):
        # Use the average gradient for all probe positions
        return np.mean(
            op.grad_probe(data, psi, scan, probe),
            axis=(1, 2),
            keepdims=True,
        )

    probe, cost = conjugate_gradient(
        None,
        x=probe,
        cost_function=cost_function,
        grad=grad,
        num_iter=num_iter,
    )

    logger.info('%10s cost is %+12.5e', 'probe', cost)
    return probe, cost


def update_object(op, data, psi, scan, probe, num_iter=1):
    """Solve the object recovery problem."""

    def cost_function(psi):
        return op.cost(data, psi, scan, probe)

    def grad(psi):
        return op.grad(data, psi, scan, probe)

    psi, cost = conjugate_gradient(
        None,
        x=psi,
        cost_function=cost_function,
        grad=grad,
        num_iter=num_iter,
    )

    logger.info('%10s cost is %+12.5e', 'object', cost)
    return psi, cost
