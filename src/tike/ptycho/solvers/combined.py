import logging

from tike.opt import conjugate_gradient, line_search
from ..position import update_positions_pd

logger = logging.getLogger(__name__)


def combined(
    op,
    num_gpu, data, probe, scan, psi,
    recover_psi=True, recover_probe=True, recover_positions=False,
    cg_iter=4,
    **kwargs
):  # yapf: disable
    """Solve the ptychography problem using a combined approach.

    """
    if recover_psi:
        psi, cost = update_object(op, num_gpu, data, psi, scan, probe, num_iter=cg_iter)

    if recover_probe:
        probe, cost = update_probe(op, num_gpu, data, psi, scan, probe, num_iter=cg_iter)

    if recover_positions:
        scan, cost = update_positions_pd(op, data, psi, probe, scan)

    return {'psi': psi, 'probe': probe, 'cost': cost, 'scan': scan}


def update_probe(op, num_gpu, data, psi, scan, probe, num_iter=1):
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


def update_object(op, num_gpu, data, psi, scan, probe, num_iter=1):
    """Solve the object recovery problem."""

    def cost_function(psi):
        return op.cost(data, psi, scan, probe)

    def grad(psi):
        return op.grad(data, psi, scan, probe)

    def cost_function_multi(psi, **kwargs):
        return op.cost_multi(num_gpu, data, psi, scan, probe, **kwargs)

    def grad_multi(psi):
        return op.grad_multi(num_gpu, data, psi, scan, probe)

    def update_multi(psi, *args):
        return op.update_multi(num_gpu, psi, *args)

    if (num_gpu<=1):
        psi, cost = conjugate_gradient(
            op.xp,
            x=psi,
            cost_function=cost_function,
            grad=grad,
            mGPU=False,
            num_iter=num_iter,
        )
    else:
        psi, cost = conjugate_gradient(
            op.xp,
            x=psi,
            cost_function=cost_function_multi,
            grad=grad_multi,
            update=update_multi,
            mGPU=True,
            num_iter=num_iter,
        )


    logger.info('%10s cost is %+12.5e', 'object', cost)
    return psi, cost
