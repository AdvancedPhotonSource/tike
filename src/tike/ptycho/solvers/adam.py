import logging

import cupy as cp
import numpy as np

import tike.linalg
from tike.opt import batch_indicies, get_batch, adam, put_batch
from ..position import update_positions_pd, PositionOptions
from ..object import positivity_constraint, smoothness_constraint
from ..probe import constrain_variable_probe, get_varying_probe
from tike.pca import pca_svd

logger = logging.getLogger(__name__)


def adam_grad(
    op, comm,
    data, probe, scan, psi,
    cost=None,
    eigen_probe=None,
    eigen_weights=None,
    num_batch=1,
    subset_is_random=True,
    probe_options=None,
    position_options=None,
    object_options=None,
):  # yapf: disable
    """Solve the ptychography problem using ADAptive Moment gradient descent.

    Parameters
    ----------
    op : :py:class:`tike.operators.Ptycho`
        A ptychography operator.
    comm : :py:class:`tike.communicators.Comm`
        An object which manages communications between both
        GPUs and nodes.


    .. seealso:: :py:mod:`tike.ptycho`

    """
    cost = np.inf
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
            psi, cost, object_options = _update_object(
                op,
                comm,
                bdata,
                psi,
                bscan,
                probe,
                eigen_probe=beigen_probe,
                eigen_weights=beigen_weights,
                object_options=object_options,
            )
            psi = comm.pool.map(positivity_constraint,
                                psi,
                                r=object_options.positivity_constraint)
            psi = comm.pool.map(smoothness_constraint,
                                psi,
                                a=object_options.smoothness_constraint)

        if probe_options:
            for m in list(range(probe[0].shape[-3])):
                probe, cost, probe_options = _update_probe(
                    op,
                    comm,
                    bdata,
                    psi,
                    bscan,
                    probe,
                    mode=[m],
                    probe_options=probe_options,
                    eigen_probe=beigen_probe,
                    eigen_weights=beigen_weights,
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

    if isinstance(eigen_probe, list):
        eigen_probe, eigen_weights = (list(a) for a in zip(*comm.pool.map(
            constrain_variable_probe,
            eigen_probe,
            eigen_weights,
        )))

    result = {
        'psi': psi,
        'probe': probe,
        'cost': cost,
        'scan': scan,
        'probe_options': probe_options,
        'object_options': object_options,
        'position_options': position_options,
    }
    if isinstance(eigen_probe, list):
        result['eigen_probe'] = eigen_probe
        result['eigen_weights'] = eigen_weights
    return result


def grad_probe(data, psi, scan, probe, mode=None, op=None):
    """Compute the gradient with respect to the probe(s).

        Parameters
        ----------
        mode : list(int)
            Only return the gradient with resepect to these probes.

    """
    self = op
    mode = list(range(probe.shape[-3])) if mode is None else mode
    intensity, farplane = self._compute_intensity(data, psi, scan, probe)
    # Use the average gradient for all probe positions
    gradient = self.adj_probe(
        farplane=self.propagation.grad(
            data,
            farplane[..., mode, :, :],
            intensity,
        ),
        psi=psi,
        scan=scan,
        overwrite=True,
    )
    mean_grad = self.xp.mean(
        gradient,
        axis=0,
        keepdims=True,
    )
    return mean_grad, gradient


def _update_probe(
    op,
    comm,
    data,
    psi,
    scan,
    probe,
    mode,
    probe_options,
    eigen_probe,
    eigen_weights,
    step_length=0.1,
):
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
        mgrad_list, grad_list = zip(*comm.pool.map(
            grad_probe,
            data,
            psi,
            scan,
            unique_probe,
            mode=mode,
            op=op,
        ))
        if comm.use_mpi:
            return comm.Allreduce_reduce(mgrad_list, 'gpu'), grad_list
        else:
            return comm.reduce(mgrad_list, 'gpu'), grad_list

    def dir_multi(dir):
        """Scatter dir to all GPUs"""
        return comm.pool.bcast(dir)

    def update_multi(x, gamma, d):

        def f(x, d):
            x[..., mode, :, :] = x[..., mode, :, :] - gamma * d
            return x

        return comm.pool.map(f, x, d)

    d, gradient = grad(probe)

    probe_options.use_adaptive_moment = True
    if probe_options.v is None or probe_options.m is None:
        probe_options.v = cp.zeros_like(probe[0])
        probe_options.m = cp.zeros_like(probe[0])
    (
        d,
        probe_options.v[..., mode, :, :],
        probe_options.m[..., mode, :, :],
    ) = adam(
        g=d[0],
        v=probe_options.v[..., mode, :, :],
        m=probe_options.m[..., mode, :, :],
        vdecay=probe_options.vdecay,
        mdecay=probe_options.mdecay,
    )
    d = [d]

    probe = update_multi(
        probe,
        gamma=step_length,
        d=dir_multi(d),
    )

    if eigen_probe is not None:
        residuals = comm.pool.map(cp.subtract, gradient, d)
        comm.pool.map(
            _update_eigen_modes,
            residuals,
            eigen_probe,
            eigen_weights,
            m=mode,
        )

    if probe[0].shape[-3] > 1 and probe_options.orthogonality_constraint:
        probe = comm.pool.map(tike.linalg.orthogonalize_gs,
                              probe,
                              axis=(-2, -1))

    cost = cost_function(probe)

    logger.info('%10s cost is %+12.5e', 'probe', cost)
    return probe, cost, probe_options


def _update_eigen_modes(residuals, eigen_probe, eigen_weights, m, alpha=0.5):
    residuals = cp.moveaxis(residuals, -5, -3)
    residuals = residuals.reshape(*residuals.shape[:-2], -1)
    W, C = pca_svd(residuals, k=eigen_probe.shape[-4])
    C = cp.moveaxis(C, -2, -3)
    C = C.reshape(*C.shape[:-1], *eigen_probe.shape[-2:])
    W = cp.moveaxis(W[0], -2, -3)
    W = cp.moveaxis(W, -1, -2)
    eigen_probe[...,
                m, :, :] = (1 - alpha) * eigen_probe[..., m, :, :] + alpha * C
    eigen_weights[..., 1:,
                  m] = (1 - alpha) * eigen_weights[..., 1:, m] + alpha * W


def _update_object(
    op,
    comm,
    data,
    psi,
    scan,
    probe,
    object_options,
    eigen_probe,
    eigen_weights,
    step_length=0.1,
):
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

    d = -grad_multi(psi)[0]

    object_options.use_adaptive_moment = True
    d, object_options.v, object_options.m = adam(
        g=d,
        v=object_options.v,
        m=object_options.m,
        vdecay=object_options.vdecay,
        mdecay=object_options.mdecay,
    )
    d = [d]

    psi = update_multi(
        psi,
        gamma=step_length,
        dir=dir_multi(d),
    )

    cost = cost_function_multi(psi)

    logger.info('%10s cost is %+12.5e', 'object', cost)
    return psi, cost, object_options
