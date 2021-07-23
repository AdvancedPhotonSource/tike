import logging

import tike.linalg
from tike.opt import conjugate_gradient, line_search

logger = logging.getLogger(__name__)


def _estimate_step_length(obj, fwd_data, theta, grid, op):
    """Use norm of forward adjoint operations to estimate step length.

    Scaling the adjoint operation by |F*Fm| / |m| puts the step length in the
    proper order of magnitude.

    """
    logger.info('Estimate step length from forward adjoint operations.')
    outnback = op.adj(
        data=fwd_data,
        theta=theta,
        grid=grid,
        overwrite=False,
    )
    scaler = tike.linalg.norm(outnback) / tike.linalg.norm(obj)
    # Multiply by 2 to because we prefer over-estimating the step
    return 2 * scaler if op.xp.isfinite(scaler) else 1.0


def cgrad(
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
        return comm.pool.grouped_allreduce(fwd_data, obj_split)

    fwd_data = fwd_op(obj)
    step_length = comm.pool.reduce_cpu(
        comm.pool.map(
            _estimate_step_length,
            obj,
            fwd_data,
            theta,
            grid,
            op=op,
        )) / comm.pool.num_workers if step_length == 1 else step_length
    exit()

    obj, cost = update_obj(
        op,
        comm,
        data,
        theta,
        obj,
        grid,
        fwd_op=fwd_op,
        num_iter=cg_iter,
        step_length=step_length,
    )

    return {'obj': obj, 'cost': cost, 'step_length': step_length}


def update_obj(op, comm, data, theta, obj, fwd_op, num_iter=1, step_length=1):
    """Solver the object recovery problem."""

    def cost_function(obj):
        cost_out = comm.pool.map(op.cost, data, theta, obj)
        if comm.use_mpi:
            return comm.Allreduce_reduce(cost_out, 'cpu')
        else:
            return comm.reduce(cost_out, 'cpu')

    def grad(obj):
        grad_list = comm.pool.map(op.grad, data, theta, obj)
        if comm.use_mpi:
            return comm.Allreduce_reduce(grad_list, 'gpu')
        else:
            return comm.reduce(grad_list, 'gpu')

    def dir_multi(dir):
        """Scatter dir to all GPUs"""
        return comm.pool.bcast(dir)

    def update_multi(x, gamma, dir):

        def f(x, dir):
            return x + gamma * dir

        return comm.pool.map(f, x, dir)

    obj, cost = conjugate_gradient(
        op.xp,
        x=obj,
        cost_function=cost_function,
        grad=grad,
        dir_multi=dir_multi,
        update_multi=update_multi,
        num_iter=num_iter,
        step_length=step_length,
    )

    logger.info('%10s cost is %+12.5e', 'object', cost)
    return obj, cost
