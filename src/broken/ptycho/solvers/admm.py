import logging

from tike.opt import conjugate_gradient, line_search

logger = logging.getLogger(__name__)


def admm1(
    op,
    data, probe, scan, psi, nearplane=None, farplane=None,
    ρ=0.5, λ=0, τ=0.5, μ=0,
    recover_psi=True, recover_probe=True, recover_nearplane=False,
    cg_iter=4,
    **kwargs
):  # yapf: disable
    return admm(
        op,
        data, probe, scan, psi, nearplane, farplane,
        ρ, λ, τ, μ,
        recover_psi, recover_probe, recover_nearplane,
        cg_iter,
        **kwargs
    )  # yapf: disable


def admm(
    op,
    data, probe, scan, psi, nearplane=None, farplane=None,
    ρ=0.5, λ=0, τ=0.5, μ=0,
    recover_psi=True, recover_probe=True, recover_nearplane=True,
    cg_iter=4,
    **kwargs
):  # yapf: disable
    """Solve the ptychography problem using ADMM.

    References
    ----------
    Selin Aslan

    """
    if nearplane is None:
        nearplane = op.diffraction.fwd(psi=psi, scan=scan, probe=probe)
    if farplane is None:
        farplane = op.propagation.fwd(nearplane)

    farplane, cost = update_phase(op, data, farplane, nearplane, ρ, λ,
                                  num_iter=cg_iter)  # yapf: disable

    if recover_nearplane:
        nearplane, cost = update_nearplane(
            op, nearplane, farplane, probe, psi, scan,
            ρ, λ, τ, μ, num_iter=cg_iter,
        )  # yapf: disable
    else:
        nearplane = op.propagation.adj(farplane)

    if recover_psi:
        psi, cost = update_object(op, nearplane, probe, scan, psi, μ, τ,
                                  num_iter=cg_iter)  # yapf: disable

    if recover_probe:
        probe, cost = update_probe(op, nearplane, probe, scan, psi, μ, τ,
                                   num_iter=cg_iter)  # yapf: disable

    λ = λ + ρ * (op.propagation.fwd(nearplane) - farplane)
    μ = μ + τ * (op.diffraction.fwd(probe=probe, psi=psi, scan=scan) -
                 nearplane)

    return {
        'psi': psi,
        'probe': probe,
        'cost': cost,
        'scan': scan,
        'λ': λ,
        'μ': μ,
        'nearplane': nearplane,
        'farplane': farplane,
    }


def update_phase(op, data, farplane, nearplane, ρ, λ, num_iter=1):
    """Solve the farplane phase problem."""
    xp = op.xp
    farplane0 = op.propagation.fwd(nearplane)

    def cost_function(farplane):
        return (op.propagation.cost(data, farplane) +
                ρ * xp.linalg.norm(xp.ravel(farplane0 - farplane + λ / ρ))**2)

    def grad(farplane):
        return (op.propagation.grad(data, farplane) - ρ *
                (farplane0 - farplane + λ / ρ))

    farplane, cost = conjugate_gradient(
        op.xp,
        x=farplane,
        cost_function=cost_function,
        grad=grad,
        num_iter=num_iter,
    )

    # print cost function for sanity check
    logger.info('%10s cost is %+12.5e', 'farplane', cost)
    return farplane, cost


def update_nearplane(
    op, nearplane, farplane, probe, psi, scan,
    ρ, λ, τ, μ, num_iter=1,
):  # yapf: disable
    """Solve the nearplane problem."""
    xp = op.xp
    nearplane0 = op.diffraction.fwd(probe=probe, psi=psi, scan=scan)

    def cost_function(nearplane):
        return (
            + ρ * xp.linalg.norm(xp.ravel(
                + op.propagation.fwd(nearplane)
                - farplane
                + λ / ρ
            ))**2
            + τ * xp.linalg.norm(xp.ravel(
                + nearplane0
                - nearplane
                + μ / τ
            ))**2
        )  # yapf: disable

    def grad(nearplane):
        return (
            + ρ * op.propagation.adj(
                + op.propagation.fwd(nearplane)
                - farplane
                + λ / ρ
            )
            - τ * (
                + nearplane0
                - nearplane
                + μ / τ
            )
        )  # yapf: disable

    nearplane, cost = conjugate_gradient(
        op.xp,
        x=nearplane,
        cost_function=cost_function,
        grad=grad,
        num_iter=num_iter,
    )

    # print cost function for sanity check
    logger.info('%10s cost is %+12.5e', 'nearplane', cost)
    return nearplane, cost


def update_probe(op, nearplane, probe, scan, psi, μ, τ, num_iter=1):
    """Solve the nearplane single probe recovery problem."""
    xp = op.xp
    obj_patches = op.diffraction.fwd(psi=psi,
                                     scan=scan,
                                     probe=xp.ones_like(probe))

    def cost_function(probe):
        return τ * xp.linalg.norm(
            xp.ravel(probe * obj_patches - nearplane + μ / τ))**2

    def grad(probe):
        # Use the average gradient for all probe positions
        return xp.mean(
            τ * xp.conj(obj_patches) *
            (probe * obj_patches - nearplane + μ / τ),
            axis=(1, 2),
            keepdims=True,
        )

    probe, cost = conjugate_gradient(
        op.xp,
        x=probe,
        cost_function=cost_function,
        grad=grad,
        num_iter=num_iter,
    )

    logger.info('%10s cost is %+12.5e', 'probe', cost)
    return probe, cost


def update_object(op, nearplane, probe, scan, psi, μ, τ, num_iter=1):
    """Solve the nearplane object recovery problem."""
    xp = op.xp

    def cost_function(psi):
        return τ * xp.linalg.norm(
            xp.ravel(
                op.diffraction.fwd(psi=psi, scan=scan, probe=probe) -
                nearplane + μ / τ))**2

    def grad(psi):
        return τ * op.diffraction.adj(
            nearplane=(op.diffraction.fwd(psi=psi, scan=scan, probe=probe) -
                       nearplane + μ / τ),
            scan=scan,
            probe=probe,
        )

    psi, cost = conjugate_gradient(
        op.xp,
        x=psi,
        cost_function=cost_function,
        grad=grad,
        num_iter=num_iter,
    )

    logger.info('%10s cost is %+12.5e', 'object', cost)
    return psi, cost
