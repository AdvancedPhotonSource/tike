import logging

from tike.linalg import orthogonalize_gs
from tike.opt import conjugate_gradient, get_batch, randomizer
from ..position import update_positions_pd, PositionOptions
from ..object import positivity_constraint, smoothness_constraint

logger = logging.getLogger(__name__)


def cgrad(
    op,
    comm,
    data,
    batches,
    *,
    parameters,
):
    """Solve the ptychography problem using conjugate gradient.

    Parameters
    ----------
    op : :py:class:`tike.operators.Ptycho`
        A ptychography operator.
    comm : :py:class:`tike.communicators.Comm`
        An object which manages communications between GPUs and nodes.
    data : list((FRAME, WIDE, HIGH) float32, ...)
        A list of unique CuPy arrays for each device containing
        the intensity (square of the absolute value) of the propagated
        wavefront; i.e. what the detector records. FFT-shifted so the
        diffraction peak is at the corners.
    batches : list(list((BATCH_SIZE, ) int, ...), ...)
        A list of list of indices along the FRAME axis of `data` for
        each device which define the batches of `data` to process
        simultaneously.
    parameters : :py:class:`tike.ptycho.solvers.PtychoParameters`
        An object which contains reconstruction parameters.

    Returns
    -------
    result : dict
        A dictionary containing the updated keyword-only arguments passed to
        this function.

    .. seealso:: :py:mod:`tike.ptycho`

    """
    probe = parameters.probe
    scan = parameters.scan
    psi = parameters.psi
    algorithm_options = parameters.algorithm_options
    probe_options = parameters.probe_options
    position_options = parameters.position_options
    object_options = parameters.object_options
    batch_cost = []

    for n in randomizer.permutation(len(batches[0])):

        bdata = comm.pool.map(get_batch, data, batches, n=n)
        bscan = comm.pool.map(get_batch, scan, batches, n=n)

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
                num_iter=algorithm_options.cg_iter,
                step_length=algorithm_options.step_length,
                object_options=object_options,
            )
            psi = comm.pool.map(positivity_constraint,
                                psi,
                                r=object_options.positivity_constraint)
            psi = comm.pool.map(smoothness_constraint,
                                psi,
                                a=object_options.smoothness_constraint)

        if probe_options:
            probe, cost, probe_options = _update_probe(
                op,
                comm,
                bdata,
                psi,
                bscan,
                probe,
                num_iter=algorithm_options.cg_iter,
                step_length=algorithm_options.step_length,
                mode=list(range(probe[0].shape[-3])),
                probe_options=probe_options,
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

        batch_cost.append(cost)

    algorithm_options.costs.append(batch_cost)
    parameters.probe = probe
    parameters.psi = psi
    parameters.scan = scan
    parameters.algorithm_options = algorithm_options
    parameters.probe_options = probe_options
    parameters.object_options = object_options
    parameters.position_options = position_options
    return parameters


def _update_probe(
    op,
    comm,
    data,
    psi,
    scan,
    probe,
    num_iter,
    step_length,
    mode,
    probe_options,
):
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
        step_length=step_length,
    )

    if probe[0].shape[-3] > 1 and probe_options.orthogonality_constraint:
        probe = comm.pool.map(orthogonalize_gs, probe, axis=(-2, -1))

    logger.info('%10s cost is %+12.5e', 'probe', cost)
    return probe, cost, probe_options


def _update_object(
    op,
    comm,
    data,
    psi,
    scan,
    probe,
    num_iter,
    step_length,
    object_options,
):
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
    return psi, cost, object_options
