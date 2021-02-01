import logging

import numpy as np

from tike.opt import conjugate_gradient
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
        )

    if recover_probe:
        # TODO: add multi-GPU support
        probe, cost = _update_probe(
            op,
            comm,
            comm.pool.gather(data, axis=1),
            psi[0],
            comm.pool.gather(scan, axis=1),
            probe[0],
            num_iter=cg_iter,
        )
        probe = comm.pool.bcast(probe)

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


def _update_probe(op, comm, data, psi, scan, probe, num_iter=1):
    """Solve the probe recovery problem."""

    def cost_function(mode):
        intensity, _ = op._compute_intensity(op, psi, scan, mode)
        return op.propagation.cost(data, intensity)

    def grad(mode):
        intensity, farplane = op._compute_intensity(op, psi, scan, mode)
        # Use the average gradient for all probe positions
        return op.xp.mean(
            op.adj_probe(
                farplane=op.propagation.grad(
                    data,
                    farplane,
                    intensity,
                ),
                psi=psi,
                scan=scan,
                overwrite=True,
            ),
            axis=1,
            keepdims=True,
        )

    probe, cost = conjugate_gradient(
        op.xp,
        x=probe,
        cost_function=cost_function,
        grad=grad,
        num_iter=num_iter,
        step_length=1,
    )

    logger.info('%10s cost is %+12.5e', 'probe', cost)
    return probe, cost


def _update_object(op, comm, data, psi, scan, probe, num_iter=1):
    """Solve the object recovery problem."""

    def cost_function_multi(psi, **kwargs):
        cost_out = comm.pool.map(op.cost, data, psi, scan, probe)
        if comm.use_mpi:
            return comm.Allreduce_reduce(cost_out, 'cpu')
        else:
            return comm.reduce(cost_out, 'cpu')

    def grad_multi(psi):
        grad_out = comm.pool.map(op.grad, data, psi, scan, probe)
        grad_list = list(grad_out)
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
        step_length=1,
    )

    logger.info('%10s cost is %+12.5e', 'object', cost)
    return psi, cost
