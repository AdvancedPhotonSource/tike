import logging

from tike.opt import conjugate_gradient, line_search

logger = logging.getLogger(__name__)


def cgrad(
    op,
    data, obj,
    cg_iter=4,
    **kwargs
):  # yapf: disable
    """Solve the Laminogarphy problem using the conjugate gradients method."""

    obj, cost = update_obj(op, data, obj, num_iter=cg_iter)

    return {'obj': obj, 'cost': cost}


def update_obj(op, data, obj, num_iter=1):
    """Solver the object recovery problem."""

    def cost_function(obj):
        "Cost function for the least-squres laminography problem"
        return op.xp.linalg.norm((op.fwd(obj) - data).ravel())**2

    def grad(obj):
        "Gradient for the least-squares laminography problem"
        return op.adj(data=(op.fwd(obj) - data)) / (op.ntheta * op.n**3)

    obj, cost = conjugate_gradient(
        op.xp,
        x=obj,
        cost_function=cost_function,
        grad=grad,
        num_iter=num_iter,
    )

    logger.info('%10s cost is %+12.5e', 'object', cost)
    return obj, cost
