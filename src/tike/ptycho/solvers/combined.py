import logging

import numpy as np

from tike.opt import conjugate_gradient, line_search

logger = logging.getLogger(__name__)

test test test
def combined(
    operator,
    data, probe, scan, psi,
    recover_psi=True, recover_probe=True,
    **kwargs,
):  # yapf: disable
    """Solve the ptychography problem using a combined approach.

    .. seealso:: tike.ptycho.divided
    """
    if recover_psi:

        def cost_psi(psi):
            return operator.cost(data, psi, scan, probe)

        def grad_psi(psi):
            return operator.grad(data, psi, scan, probe)

        psi, cost = conjugate_gradient(
            operator.array_module,
            x=psi,
            cost_function=cost_psi,
            grad=grad_psi,
            num_iter=2,
        )

    return {
        'psi': psi,
        'probe': probe,
        'cost': cost,
    }
