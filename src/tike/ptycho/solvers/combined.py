import logging

import numpy as np

from tike.opt import conjugate_gradient, line_search, direction_dy
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


def _update_probe(op, comm, data, psi, scan, probe, num_iter, step_length):
    """Solve the probe recovery problem."""

    def cost_function(probe):
        cost_out = comm.pool.map(op.cost, data, psi, scan, probe)
        if comm.use_mpi:
            return comm.Allreduce_reduce(cost_out, 'cpu')
        else:
            return comm.reduce(cost_out, 'cpu')

    def grad(probe):
        grad_list = comm.pool.map(op.grad_probe, data, psi, scan, probe)
        if comm.use_mpi:
            return comm.Allreduce_reduce(grad_list, 'gpu')
        else:
            return comm.reduce(grad_list, 'gpu')

    def dir_multi(dir):
        """Scatter dir to all GPUs"""
        return comm.pool.bcast(dir)

    def update_multi(x, gamma, d):

        def f(x, d):
            return x + gamma * d

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


def _update_both(op, comm, data, psi, scan, probe, num_iter=1):
    """Update all of the parameters simultaneously.

    In tests with experimental data, this approach does not converge as fast as
    the sequential update approach. This is probably because there is too much
    ambiguity / redundancy in the parameters. For example, the gradients of
    each of the multiple-probes is identical (they each recieve the same
    gradient) if they are seeded identically, so they are completely
    interchangeable. Additionally, either the object and probe can be scaled
    arbitrarily.

    """

    def cost_function(params):
        """Gather the objective value from all workers.

        Parameters
        ----------
        params : [[x0, x1, x2, ...], [y0, y1, y2, ...], ...]
            A list of parameters on each worker.

        """
        psi, probe = params
        cost_out = comm.pool.map(op.cost, data, psi, scan, probe)
        if comm.use_mpi:
            return comm.Allreduce_reduce(cost_out, 'cpu')
        else:
            return comm.reduce(cost_out, 'cpu')

    def gradient(params):
        """Gather the gradient from all workers to one worker.

        Parameters
        ----------
        params : [[x0, x1, x2, ...], [y0, y1, y2, ...], ...]
            A list of parameters on each worker.

        Returns
        -------
        grads : [dx0, dy0]
            A list of gradients for each parameter on one worker.
        """
        psi, probe = params
        gradients = zip(*comm.pool.map(op.grad, data, psi, scan, probe))
        grad_both = []
        for grad_list in gradients:
            if comm.use_mpi:
                grad_both.append(comm.Allreduce_reduce(grad_list, 'gpu'))
            else:
                grad_both.append(comm.reduce(grad_list, 'gpu'))
        return grad_both

    def dir_multi(directions):
        """Scatter the new search direction to all workers.

        Parameters
        ----------
        directions : [dx0, dy0, dz0, ...]
            A list of direction on a single worker.

        Returns
        -------
        directions: [[dx0, dx1, dx2,...], [dy0, dy1, dy3, ...], ...]
            A list of broadcast directions on each worker.

        """
        return [comm.pool.bcast(d) for d in directions]

    def update_multi(params, gamma, directions):
        """Update both along the search direction.

        Parameters
        ----------
        directions : [[dx0, dx1, dx2,...], [dy0, dy1, dy3, ...], ...]
            A list of search directions on each worker.
        gamma : float
            A step length
        params : [[x0, x1, x2, ...], [y0, y1, y2, ...], ...]
            A list of parameters on each worker.

        Returns
        -------
        params : [[x0, x1, x2, ...], [y0, y1, y2, ...], ...]
            A list of the updated parameters on each worker.

        """

        def f(x, d):
            return x + gamma * d

        return [comm.pool.map(f, p, d) for p, d in zip(params, directions)]

    array_module = op.xp
    x = (psi, probe)
    grad = gradient
    step_length = 1e-3

    for i in range(num_iter):

        grad1 = grad(x)
        if i == 0:
            dir_ = [-g for g in grad1]
        else:
            dir_ = [
                direction_dy(array_module, g0, g1, d)
                for g0, g1, d in zip(grad0, grad1, dir_)
            ]
        grad0 = grad1

        dir_list = dir_multi(dir_)

        gamma, cost = line_search(
            f=cost_function,
            x=x,
            d=dir_list,
            update_multi=update_multi,
            step_length=step_length,
        )

        x = update_multi(x, gamma, dir_list)

        logger.debug("step %d; length %.3e -> %.3e; cost %.6e", i, step_length,
                     gamma, cost)

    logger.info('%10s cost is %+12.5e', 'object', cost)
    return (*x, cost)
