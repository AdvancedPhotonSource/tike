import logging

from tike.opt import conjugate_gradient

logger = logging.getLogger(__name__)


def cgrad(
    op,
    original,
    unaligned,
    reg, rho_p, rho_a,
    num_iter=4,
    cost=None,
    **kwargs
):  # yapf: disable
    """Recover an undistorted image from a given flow."""

    def cost_function(original):
        # yapf: disable
        return (
            rho_p * op.xp.linalg.norm(op.xp.ravel(
                unaligned - op.fwd(
                    original,
                    padded_shape=unaligned.shape,
                    **kwargs,
            )))**2 +
            rho_a * op.xp.linalg.norm(op.xp.ravel(
                original - reg
            ))**2
        )
        # yapf: enable

    def grad(original):
        return (rho_p * op.adj(
            op.fwd(
                original,
                padded_shape=unaligned.shape,
                **kwargs,
            ) - unaligned,
            unpadded_shape=original.shape,
            **kwargs,
        ) + rho_a * (original - reg))

    original, cost = conjugate_gradient(
        op.xp,
        x=original,
        cost_function=cost_function,
        grad=grad,
        num_iter=num_iter,
    )

    logger.info('%10s cost is %+12.5e', 'original', cost)
    return {'original': original, 'cost': cost}
