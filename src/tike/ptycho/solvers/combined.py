import logging

import numpy as np

from tike.opt import conjugate_gradient, line_search
from ..position import update_positions_pd

logger = logging.getLogger(__name__)


def combined(
    op,
    pool,
    data, probe, scan, psi,
    recover_psi=True, recover_probe=True, recover_positions=False,
    cg_iter=4,
):  # yapf: disable
    """Solve the ptychography problem using a combined approach.

    Parameters
    ----------
    operator : tike.operators.Ptycho
        A ptychography operator.
    pool : tike.pool.ThreadPoolExecutor
        An object which manages communications between GPUs.
    """
    cost = np.inf

    if recover_psi:
        psi, cost = update_object(
            op,
            pool,
            data,
            psi,
            scan,
            probe,
            num_iter=cg_iter,
        )

    if recover_probe:
        # TODO: add multi-GPU support
        probe, cost = update_probe(
            op,
            pool,
            pool.gather(data, axis=1),
            psi[0],
            pool.gather(scan, axis=1),
            probe[0],
            num_iter=cg_iter,
        )
        probe = pool.bcast(probe)

    if recover_positions and pool.num_workers == 1:
        scan, cost = update_positions_pd(
            op,
            pool.gather(data, axis=1),
            psi[0],
            probe[0],
            pool.gather(scan, axis=1),
        )
        scan = pool.bcast(scan)

    return {'psi': psi, 'probe': probe, 'cost': cost, 'scan': scan}


def update_probe(op, pool, data, psi, scan, probe, num_iter=1):
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
            step_length=4,
        )

    logger.info('%10s cost is %+12.5e', 'probe', cost)
    return probe, cost


def update_object(op, pool, data, psi, scan, probe, num_iter=1):
    """Solve the object recovery problem."""

    def cost_function_multi(psi, **kwargs):
        cost_out = pool.map(op.cost, data, psi, scan, probe)
        # TODO: Implement reduce function for ThreadPool
        cost_cpu = 0
        for c in cost_out:
            cost_cpu += op.asnumpy(c)
        return cost_cpu

    def grad_multi(psi):
        grad_out = pool.map(op.grad, data, psi, scan, probe)
        grad_list = list(grad_out)
        # TODO: Implement reduce function for ThreadPool
        for i in range(1, len(grad_list)):
            grad_cpu_tmp = op.asnumpy(grad_list[i])
            grad_tmp = op.asarray(grad_cpu_tmp)
            grad_list[0] += grad_tmp

        return grad_list[0]

    def dir_multi(dir):
        """Scatter dir to all GPUs"""
        return pool.bcast(dir)

    def update_multi(psi, gamma, dir):

        def f(psi, dir):
            return psi + gamma * dir

        return list(pool.map(f, psi, dir))

    psi, cost = conjugate_gradient(
        op.xp,
        x=psi,
        cost_function=cost_function_multi,
        grad=grad_multi,
        dir_multi=dir_multi,
        update_multi=update_multi,
        num_iter=num_iter,
        step_length=8e-5,
    )

    logger.info('%10s cost is %+12.5e', 'object', cost)
    return psi, cost
