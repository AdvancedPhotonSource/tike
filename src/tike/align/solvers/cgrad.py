import logging

from tike.opt import conjugate_gradient

logger = logging.getLogger(__name__)


def cgrad(
    op,
    original,
    unaligned,
    flow,
    num_iter=4,
    reg=0, rho=0,
    **kwargs
):  # yapf: disable
    """Recover an undistorted image from a given flow."""

    def cost_function(original):
        return (
            0.5 * op.xp.linalg.norm((op.fwd(original, flow) - unaligned).ravel())**2 +
            0.5 * op.xp.linalg.norm((original - reg).ravel())**2
        )

    def grad(original):
        return (
            op.adj(op.fwd(original, flow) - unaligned, flow) +
            (original - reg)
        )

    cost = 0
    original, cost = conjugate_gradient(
        op.xp,
        x=original,
        cost_function=cost_function,
        grad=grad,
        num_iter=num_iter,
    )

    logger.info('%10s cost is %+12.5e', 'original', cost)
    return {'original': original, 'cost': cost}
