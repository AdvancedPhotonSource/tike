import logging

import numpy as np

from tike.opt import conjugate_gradient, line_search, direction_dy
from tike.linalg import orthogonalize_gs

from ..position import update_positions_pd

logger = logging.getLogger(__name__)


def cgrad(
    op, comm,
    data, probe, scan, psi,
    recover_psi=True, recover_probe=True, recover_positions=False,
    cg_iter=4,
    cost=None,
    eigen_probe=None,
    eigen_weights=None,
    step_length=1,
    probe_is_orthogonal=False,
):  # yapf: disable
    """Solve the ptychography problem using conjugate gradient.

    Parameters
    ----------
    op : tike.operators.Ptycho
        A ptychography operator.
    comm : tike.communicators.Comm
        An object which manages communications between both
        GPUs and nodes.

    """
    cost = np.inf

    if recover_psi:
        psi, cost = _update_object(
            op,
            comm,
            data,
            psi,
            scan,
            probe,
            num_iter=cg_iter,
            step_length=step_length,
        )

    if recover_probe:
        probe, cost = _update_probe(
            op,
            comm,
            data,
            psi,
            scan,
            probe,
            num_iter=cg_iter,
            step_length=step_length,
            probe_is_orthogonal=probe_is_orthogonal,
            mode=list(range(probe[0].shape[-3])),
        )

    if recover_positions and comm.pool.num_workers == 1:
        scan, cost = update_positions_pd(
            op,
            comm.pool.gather(data, axis=1),
            psi[0],
            probe[0],
            comm.pool.gather(scan, axis=1),
        )
        scan = comm.pool.bcast(scan)

    return {'psi': psi, 'probe': probe, 'cost': cost, 'scan': scan}


def _update_probe(op, comm, data, psi, scan, probe, num_iter, step_length,
                  probe_is_orthogonal, mode):
    """Solve the probe recovery problem."""

    def cost_function(probe):
        cost_out = comm.pool.map(op.cost, data, psi, scan, probe)
        if comm.use_mpi:
            return comm.Allreduce_reduce(cost_out, 'cpu')
        else:
            return comm.reduce(cost_out, 'cpu')

    def grad(probe):
        grad_list = comm.pool.map(
            op.grad_probe,
            data,
            psi,
            scan,
            probe,
            mode=mode,
        )
        if comm.use_mpi:
            return comm.Allreduce_reduce(grad_list, 'gpu')
        else:
            return comm.reduce(grad_list, 'gpu')

    def dir_multi(dir):
        """Scatter dir to all GPUs"""
        return comm.pool.bcast(dir)

    def update_multi(x, gamma, d):

        def f(x, d):
            return x[..., mode, :, :] + gamma * d

        return comm.pool.map(f, x, d)

    probe, cost = conjugate_gradient(
        op.xp,
        x=probe,
        cost_function=cost_function,
        grad=grad,
        dir_multi=dir_multi,
        update_multi=update_multi,
        num_iter=num_iter,
        step_length=1,
    )

    if probe[0].shape[-3] > 1 and probe_is_orthogonal:
        probe = comm.pool.map(orthogonalize_gs, probe, axis=(-2, -1))

    logger.info('%10s cost is %+12.5e', 'probe', cost)
    return probe, cost


def _update_object(op, comm, data, psi, scan, probe, num_iter, step_length):
    """Solve the object recovery problem."""

    def cost_function_multi(psi, **kwargs):
        cost_out = comm.pool.map(op.cost, data, psi, scan, probe)
        if comm.use_mpi:
            return comm.Allreduce_reduce(cost_out, 'cpu')
        else:
            return comm.reduce(cost_out, 'cpu')

    def grad_multi(psi):
        grad_list = comm.pool.map(op.grad_psi, data, psi, scan, probe)
        if comm.use_mpi:
            return comm.Allreduce_reduce(grad_list, 'gpu')
        else:
            return comm.reduce(grad_list, 'gpu')

    def dir_multi(dir):
        """Scatter dir to all GPUs"""
        return comm.pool.bcast(dir)

    def update_multi(psi, gamma, dir):

        def f(psi, dir):
            return psi + gamma * dir

        return list(comm.pool.map(f, psi, dir))

    psi, cost = conjugate_gradient(
        op.xp,
        x=psi,
        cost_function=cost_function_multi,
        grad=grad_multi,
        dir_multi=dir_multi,
        update_multi=update_multi,
        num_iter=num_iter,
        step_length=step_length,
    )

    logger.info('%10s cost is %+12.5e', 'object', cost)
    return psi, cost
