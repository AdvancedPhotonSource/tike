import logging

from tike.opt import conjugate_gradient, line_search
from ..position import update_positions_pd

logger = logging.getLogger(__name__)


def combined(
    op,
    pool,
    num_gpu, data, probe, scan, psi,
    recover_psi=True, recover_probe=True, recover_positions=False,
    cg_iter=4,
    **kwargs
):  # yapf: disable
    """Solve the ptychography problem using a combined approach.

    Parameters
    ----------
    operator : tike.operators.Ptycho
        A ptychography operator.
    pool : tike.pool.ThreadPoolExecutor
        An object which manages communications between GPUs.
    """
    if recover_psi:
        psi, cost = update_object(
            op,
            pool,
            num_gpu,
            data,
            psi,
            scan,
            probe,
            num_iter=cg_iter,
        )

    if recover_probe:
        probe, cost = update_probe(
            op,
            pool,
            num_gpu,
            data,
            psi,
            scan,
            probe,
            num_iter=cg_iter,
        )

    if recover_positions:
        scan, cost = update_positions_pd(op, data, psi, probe, scan)

    return {'psi': psi, 'probe': probe, 'cost': cost, 'scan': scan}


def update_probe(op, pool, num_gpu, data, psi, scan, probe, num_iter=1):
    """Solve the probe recovery problem."""
    # TODO: add multi-GPU support
    if (num_gpu > 1):
        scan = pool.gather(scan, axis=1)
        data = pool.gather(data, axis=1)
        psi = psi[0]
        probe = probe[0]

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

    if (num_gpu > 1):
        probe = pool.bcast(probe)
        del scan
        del data

    logger.info('%10s cost is %+12.5e', 'probe', cost)
    return probe, cost


def update_object(op, pool, num_gpu, data, psi, scan, probe, num_iter=1):
    """Solve the object recovery problem."""

    def cost_function(psi):
        return op.cost(data, psi, scan, probe)

    def grad(psi):
        return op.grad(data, psi, scan, probe)

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
        for i in range(1, num_gpu):
            # TODO: Implement optimal reduce in ThreadPool
            # if cp.cuda.runtime.deviceCanAccessPeer(0, i):
            #     cp.cuda.runtime.deviceEnablePeerAccess(i)
            #     grad_tmp.data.copy_from_device(
            #         grad_list[i].data,
            #         grad_list[0].size * grad_list[0].itemsize,
            #     )
            # else:
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

    if (num_gpu <= 1):
        psi, cost = conjugate_gradient(
            op.xp,
            x=psi,
            cost_function=cost_function,
            grad=grad,
            num_gpu=num_gpu,
            num_iter=num_iter,
        )
    else:
        psi, cost = conjugate_gradient(
            op.xp,
            x=psi,
            cost_function=cost_function_multi,
            grad=grad_multi,
            dir_multi=dir_multi,
            update_multi=update_multi,
            num_gpu=num_gpu,
            num_iter=num_iter,
        )

    logger.info('%10s cost is %+12.5e', 'object', cost)
    return psi, cost
