import logging

import tike.linalg
from tike.opt import conjugate_gradient

logger = logging.getLogger(__name__)


def _estimate_step_length(obj, fwd_data, theta, grid, op, comm, s):
    """Use norm of forward adjoint operations to estimate step length.

    Scaling the adjoint operation by |F*Fm| / |m| puts the step length in the
    proper order of magnitude.

    """
    logger.info('Estimate step length from forward adjoint operations.')

    def reduce_norm(data, workers):
        def f(data):
            return tike.linalg.norm(data)**2
        sqr = comm.pool.map(f, data, workers=workers)
        if comm.use_mpi:
            sqr_sum = comm.Allreduce_reduce(sqr, 'cpu')
        else:
            sqr_sum = comm.reduce(sqr, 'cpu')
        return sqr_sum**0.5

    outnback = comm.pool.map(
        op.adj,
        fwd_data,
        theta,
        grid,
        overwrite=False,
    )
    comm.reduce(outnback, 'gpu', s=s)
    workers = comm.pool.workers[:s]
    objn = reduce_norm(obj, workers)
    # Multiply by 2 to because we prefer over-estimating the step
    return 2 * reduce_norm(outnback, workers) / objn if objn != 0.0 else 1.0


def bucket(
    op,
    comm,
    data, theta, obj, grid,
    obj_split=1,
    cg_iter=4,
    step_length=1,
    **kwargs
):  # yapf: disable
    """Solve the Laminogarphy problem using the conjugate gradients method."""

    def fwd_op(u):
        fwd_data = comm.pool.map(op.fwd, u, theta, grid)
        if comm.use_mpi:
            return comm.Allreduce(fwd_data, obj_split)
        else:
            return comm.pool.allreduce(fwd_data, obj_split)

    fwd_data = fwd_op(obj)
    if step_length == 1:
        step_length = _estimate_step_length(
            obj,
            fwd_data,
            theta,
            grid,
            op=op,
            comm=comm,
            s=obj_split,
        )
    else:
        step_length = step_length

    obj, cost = update_obj(
        op,
        comm,
        data,
        theta,
        obj,
        grid,
        obj_split,
        fwd_op=fwd_op,
        num_iter=cg_iter,
        step_length=step_length,
    )

    return {'obj': obj, 'cost': cost, 'step_length': step_length}


def update_obj(
    op,
    comm,
    data, theta, obj, grid,
    obj_split,
    fwd_op,
    num_iter=1,
    step_length=1,
):
    """Solver the object recovery problem."""

    def cost_function(obj):
        fwd_data = fwd_op(obj)
        workers = comm.pool.workers[::obj_split]
        cost_out = comm.pool.map(
            op.cost,
            data[::obj_split],
            fwd_data[::obj_split],
            workers=workers,
        )
        if comm.use_mpi:
            return comm.Allreduce_reduce(cost_out, 'cpu')
        else:
            return comm.reduce(cost_out, 'cpu')

    def grad(obj):
        fwd_data = fwd_op(obj)
        grad_list = comm.pool.map(op.grad, data, theta, fwd_data, grid)
        return comm.reduce(grad_list, 'gpu', s=obj_split)

    def direction_dy(xp, grad1, grad0=None, dir_=None):
        """Return the Dai-Yuan search direction."""

        def init(grad1):
            return -grad1

        def f(grad1):
            return xp.linalg.norm(grad1.ravel())**2

        def d(grad0, grad1, dir_, norm_):
            return (
                - grad1
                + dir_ * norm_
                / (xp.sum(dir_.conj() * (grad1 - grad0)) + 1e-32)
            )  # yapf: disable

        workers = comm.pool.workers[:obj_split]

        if dir_ is None:
            return comm.pool.map(init, grad1, workers=workers)

        n = comm.pool.map(f, grad1, workers=workers)
        if comm.use_mpi:
            norm_ = comm.Allreduce_reduce(n, 'cpu')
        else:
            norm_ = comm.reduce(n, 'cpu')
        return comm.pool.map(
            d,
            grad0,
            grad1,
            dir_,
            norm_=norm_,
            workers=workers,
        )

    def dir_multi(dir):
        """Scatter dir to all GPUs"""
        return comm.pool.bcast(dir, obj_split)

    def update_multi(x, gamma, dir):

        def f(x, dir):
            return x + gamma * dir

        return comm.pool.map(f, x, dir)

    obj, cost = conjugate_gradient(
        op.xp,
        x=obj,
        cost_function=cost_function,
        grad=grad,
        direction_dy=direction_dy,
        dir_multi=dir_multi,
        update_multi=update_multi,
        num_iter=num_iter,
        step_length=step_length,
    )

    logger.info('%10s cost is %+12.5e', 'object', cost)
    return obj, cost
