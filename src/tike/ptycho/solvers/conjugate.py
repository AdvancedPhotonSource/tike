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
    probe,
    scan,
    psi,
    batches,
    cg_iter=4,
    step_length=1,
    probe_options=None,
    position_options=None,
    object_options=None,
    cost=None,
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
    probe : list((1, 1, SHARED, WIDE, HIGH) complex64, ...)
        A list of duplicate CuPy arrays for each device containing
        the shared complex illumination function amongst all positions.
    scan : list((POSI, 2) float32, ...)
        A list of unique CuPy arrays for each device containing
        coordinates of the minimum corner of the probe grid for each
        measurement in the coordinate system of psi. Coordinate order
        consistent with WIDE, HIGH order.
    psi : list((WIDE, HIGH) complex64, ...)
        A list of duplicate CuPy arrays for each device containing
        the wavefront modulation coefficients of the object.
    batches : list(list((BATCH_SIZE, ) int, ...), ...)
        A list of list of indices along the FRAME axis of `data` for
        each device which define the batches of `data` to process
        simultaneously.
    position_options : :py:class:`tike.ptycho.PositionOptions`
        A class containing settings related to position correction.
    probe_options : :py:class:`tike.ptycho.ProbeOptions`
        A class containing settings related to probe updates.
    object_options : :py:class:`tike.ptycho.ObjectOptions`
        A class containing settings related to object updates.
    cost : float
        The current objective function value.
    cg_iter : int
        The number of conjugate directions to search for each update.
    step_length : float
        Scales the inital search directions before the line search.

    Returns
    -------
    result : dict
        A dictionary containing the updated inputs if they can be updated.

    .. seealso:: :py:mod:`tike.ptycho`

    """
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
                num_iter=cg_iter,
                step_length=step_length,
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
                num_iter=cg_iter,
                step_length=step_length,
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

    return {
        'psi': psi,
        'probe': probe,
        'cost': cost,
        'scan': scan,
        'probe_options': probe_options,
        'object_options': object_options,
        'position_options': position_options,
    }


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
