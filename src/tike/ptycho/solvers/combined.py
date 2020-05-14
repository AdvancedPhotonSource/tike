import logging

from tike.opt import conjugate_gradient, line_search
from ..position import update_positions_pd

logger = logging.getLogger(__name__)


def combined(
    op,
    data, probe, scan, psi,
    recover_psi=True, recover_probe=False, recover_positions=False,
    cg_iter=4,
    **kwargs
):  # yapf: disable
    """Solve the ptychography problem using a combined approach.

    """
    if recover_psi:
        psi, cost = update_object(op, data, psi, scan, probe, num_iter=cg_iter)

    if recover_probe:
        probe, cost = update_probe(op, data, psi, scan, probe, num_iter=cg_iter)

    if recover_positions:
        scan, cost = update_positions_pd(op, data, psi, probe, scan)

    return {'psi': psi, 'probe': probe, 'cost': cost, 'scan': scan}


def update_probe(op, data, psi, scan, probe, num_iter=1):
    """Solve the probe recovery problem."""
    # TODO: Cache object patche between mode updates
    for m in range(probe.shape[-3]):

        def cost_function(mode):
            return op.cost(data, psi, scan, probe, m, mode)

        def grad(mode):
            # Use the average gradient for all probe positions
            return op.xp.mean(
                op.grad_probe(data, psi, scan, probe, m, mode),
                axis=(1, 2),
                keepdims=True,
            )

        probe[..., m:m + 1, :, :], cost = conjugate_gradient(
            op.xp,
            x=probe[..., m:m + 1, :, :],
            cost_function=cost_function,
            grad=grad,
            num_iter=num_iter,
        )

    logger.info('%10s cost is %+12.5e', 'probe', cost)
    return probe, cost


def update_object(op, data, psi, scan, probe, num_iter=1):
    """Solve the object recovery problem."""
    gpu_count = 2

    def cost_function(psi):
        return op.cost(data, psi, scan, probe)

    def cost_function_multi(psi, **kwargs):
        return op.cost_multi(gpu_count, data, psi, scan, probe, **kwargs)

    def grad(psi):
        return op.grad(data, psi, scan, probe)

    def grad_multi(gpu_id):
        return op.grad_multi(gpu_id, data, psi, scan, probe)

    psi, cost = conjugate_gradient(
        op.xp,
        x=psi,
        cost_function=cost_function_multi,
        grad=grad_multi,
        num_iter=num_iter,
    )

    logger.info('%10s cost is %+12.5e', 'object', cost)
    return psi, cost
