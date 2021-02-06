import logging

from tike.opt import conjugate_gradient, line_search

logger = logging.getLogger(__name__)


def cgrad(
    op,
    comm,
    data, theta, obj,
    cg_iter=4,
    **kwargs
):  # yapf: disable
    """Solve the Laminogarphy problem using the conjugate gradients method."""

    obj, cost = update_obj(op, comm, data, theta, obj, num_iter=cg_iter)

    return {'obj': obj, 'cost': cost}


def update_obj(op, comm, data, theta, obj, num_iter=1):
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
    )

    logger.info('%10s cost is %+12.5e', 'object', cost)
    return obj, cost
