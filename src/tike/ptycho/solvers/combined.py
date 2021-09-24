import logging

import numpy as np

from tike.linalg import orthogonalize_gs
from tike.opt import conjugate_gradient, batch_indicies, get_batch, put_batch
from ..position import update_positions_pd, PositionOptions
from ..probe import get_varying_probe, opr
from ..object import positivity_constraint, smoothness_constraint

logger = logging.getLogger(__name__)


def cgrad(
    op, comm,
    data, probe, scan, psi,
    cg_iter=4,
    cost=None,
    eigen_probe=None,
    eigen_weights=None,
    num_batch=1,
    subset_is_random=True,
    step_length=1,
    probe_options=None,
    position_options=None,
    object_options=None,
):  # yapf: disable
    """Solve the ptychography problem using conjugate gradient.

    Parameters
    ----------
    op : :py:class:`tike.operators.Ptycho`
        A ptychography operator.
    comm : :py:class:`tike.communicators.Comm`
        An object which manages communications between both
        GPUs and nodes.


    .. seealso:: :py:mod:`tike.ptycho`

    """
    # Unique batch for each device
    batches = [
        batch_indicies(s.shape[-2], num_batch, subset_is_random) for s in scan
    ]
    for n in range(num_batch):

        bdata = comm.pool.map(get_batch, data, batches, n=n)
        bscan = comm.pool.map(get_batch, scan, batches, n=n)

        if isinstance(eigen_probe, list):
            beigen_weights = comm.pool.map(
                get_batch,
                eigen_weights,
                batches,
                n=n,
            )
            beigen_probe = eigen_probe
        else:
            beigen_probe = [None] * comm.pool.num_workers
            beigen_weights = [None] * comm.pool.num_workers

        if position_options:
            bposition_options = comm.pool.map(PositionOptions.split,
                                              position_options,
                                              [b[n] for b in batches])
        else:
            bposition_options = None

        if object_options:
            psi, cost = _update_object(
                op,
                comm,
                bdata,
                psi,
                bscan,
                probe,
                eigen_probe=beigen_probe,
                eigen_weights=beigen_weights,
                num_iter=cg_iter,
                step_length=step_length,
            )
            psi = comm.pool.map(positivity_constraint,
                                psi,
                                r=object_options.positivity_constraint)
            psi = comm.pool.map(smoothness_constraint,
                                psi,
                                a=object_options.smoothness_constraint)

        if probe_options:
            probe, cost = _update_probe(
                op,
                comm,
                bdata,
                psi,
                bscan,
                probe,
                eigen_probe=beigen_probe,
                eigen_weights=beigen_weights,
                num_iter=cg_iter,
                step_length=step_length,
                probe_is_orthogonal=probe_options.orthogonality_constraint,
                mode=list(range(probe[0].shape[-3])),
            )

        if position_options and comm.pool.num_workers == 1:
            bscan, cost = update_positions_pd(
                op,
                comm.pool.gather(bdata, axis=-3),
                psi[0],
                probe[0],
                comm.pool.gather(bscan, axis=-2),
            )
            bscan = comm.pool.bcast([bscan])
            # TODO: Assign bscan into scan when positions are updated

        if isinstance(eigen_probe, list):
            comm.pool.map(
                put_batch,
                beigen_weights,
                eigen_weights,
                batches,
                n=n,
            )

    return {'psi': psi, 'probe': probe, 'cost': cost, 'scan': scan}


def _update_probe(op, comm, data, psi, scan, probe, num_iter, step_length,
                  probe_is_orthogonal, mode, eigen_probe, eigen_weights):
    """Solve the probe recovery problem."""

    def cost_function(probe):
        unique_probe = comm.pool.map(
            get_varying_probe,
            probe,
            eigen_probe,
            eigen_weights,
        )
        cost_out = comm.pool.map(op.cost, data, psi, scan, unique_probe)
        if comm.use_mpi:
            return comm.Allreduce_reduce(cost_out, 'cpu')
        else:
            return comm.reduce(cost_out, 'cpu')

    def grad(probe):
        unique_probe = comm.pool.map(
            get_varying_probe,
            probe,
            eigen_probe,
            eigen_weights,
        )
        grad_list = comm.pool.map(
            op.grad_probe,
            data,
            psi,
            scan,
            unique_probe,
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
        step_length=step_length,
    )

    if probe[0].shape[-3] > 1 and probe_is_orthogonal:
        probe = comm.pool.map(orthogonalize_gs, probe, axis=(-2, -1))

    logger.info('%10s cost is %+12.5e', 'probe', cost)
    return probe, cost


def _update_object(op, comm, data, psi, scan, probe, num_iter, step_length,
                   eigen_probe, eigen_weights):
    """Solve the object recovery problem."""

    unique_probe = comm.pool.map(
        get_varying_probe,
        probe,
        eigen_probe,
        eigen_weights,
    )

    def cost_function_multi(psi, **kwargs):
        cost_out = comm.pool.map(op.cost, data, psi, scan, unique_probe)
        if comm.use_mpi:
            return comm.Allreduce_reduce(cost_out, 'cpu')
        else:
            return comm.reduce(cost_out, 'cpu')

    def grad_multi(psi):
        grad_list = comm.pool.map(op.grad_psi, data, psi, scan, unique_probe)
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


def _update_eigen_probe(op, comm, data, psi, scan, probe, eigen_probe,
                        eigen_weights, alpha):
    """Update the eigen probes and weights."""
    unique_probe = get_varying_probe(
        probe,
        eigen_probe,
        eigen_weights,
    )
    # Compute the gradient for each probe positions
    intensity, farplane = op._compute_intensity(data, psi, scan, unique_probe)
    gradients = op.adj_probe(
        farplane=op.propagation.grad(
            data,
            farplane,
            intensity,
        ),
        psi=psi,
        scan=scan,
        overwrite=True,
    )

    # Get the residual gradient for each probe position
    # TODO: Maybe subtracting this mean is not necessary because we already
    # updated the main probe. Or maybe it is because it keeps the residuals
    # zero-mean
    residuals = gradients - np.mean(gradients, axis=-5, keepdims=True)

    # Perform principal component analysis on the residual gradients
    eigen_probe, eigen_weights = opr(
        residuals,
        eigen_probe,
        eigen_weights,
        eigen_weights.shape[-2],
        alpha=alpha,
    )

    return eigen_probe, eigen_weights
