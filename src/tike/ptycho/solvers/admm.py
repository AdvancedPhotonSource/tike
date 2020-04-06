import logging

import numpy as np

from tike.opt import conjugate_gradient, line_search

logger = logging.getLogger(__name__)


def admm(
    op,
    data, probe, scan, psi,
    ρ=0.5, λ=0, τ=0.5, μ=0,
    recover_psi=True, recover_probe=True,
    **kwargs
):  # yapf: disable
    """Solve the ptychography problem using ADMM.

    References
    ----------
    Selin Aslan

    .. seealso:: tike.ptycho.combined, tike.ptycho.divided

    """
    nearplane = op.diffraction.fwd(psi=psi, scan=scan, probe=probe)
    farplane = op.propagation.fwd(nearplane)

    farplane, cost = update_phase(op, data, farplane, ρ, λ, num_iter=2)

    nearplane, cost = update_nearplane(
        op, nearplane, farplane, probe, psi, scan,
        ρ, λ, τ, μ, num_iter=2,
    )  # yapf: disable

    if recover_psi:
        psi, cost = update_object(op, nearplane, probe, scan, psi, μ, τ,
                                  num_iter=2)  # yapf: disable

    if recover_probe:
        probe, cost = update_probe(op, nearplane, probe, scan, psi, μ, τ,
                                   num_iter=2)  # yapf: disable

    λ = λ + ρ * (op.propagation.fwd(nearplane) - farplane)
    μ = μ + τ * (op.diffraction.fwd(probe=probe, psi=psi, scan=scan) -
                 nearplane)

    return {'psi': psi, 'probe': probe, 'cost': cost, 'scan': scan,
            'λ': λ, 'μ': μ}


def update_phase(op, data, farplane, ρ, λ, num_iter=1):
    """Solve the farplane phase problem."""
    farplane0 = farplane.copy()

    def cost_function(farplane):
        return (op.propagation.cost(data, farplane) +
                ρ * np.linalg.norm(farplane0 - farplane + λ / ρ)**2)

    def grad(farplane):
        return (op.propagation.grad(data, farplane) - ρ *
                (farplane0 - farplane + λ / ρ))

    farplane, cost = conjugate_gradient(
        None,
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

    def cost_function(nearplane):
        return (
            + ρ * np.linalg.norm(
                + op.propagation.fwd(nearplane=nearplane)
                - farplane
                + λ / ρ
            )**2
            + τ * np.linalg.norm(
                + op.diffraction.fwd(probe=probe, psi=psi, scan=scan)
                - nearplane
                + μ / τ
            )**2
        )  # yapf: disable

    def grad(nearplane):
        return (
            + ρ * op.propagation.adj(
                + op.propagation.fwd(nearplane)
                - farplane
                + λ / ρ
            )
            - τ * (
                + op.diffraction.fwd(probe=probe, psi=psi, scan=scan)
                - nearplane
                + μ / τ
            )
        )  # yapf: disable

    nearplane, cost = conjugate_gradient(
        None,
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
    obj_patches = op.diffraction.fwd(psi=psi,
                                     scan=scan,
                                     probe=np.ones_like(probe))

    def cost_function(probe):
        return τ * np.linalg.norm(probe * obj_patches - nearplane + μ / τ)**2

    def grad(probe):
        # Use the average gradient for all probe positions
        return np.mean(
            τ * np.conj(obj_patches) *
            (probe * obj_patches - nearplane + μ / τ),
            axis=(1, 2),
            keepdims=True,
        )

    probe, cost = conjugate_gradient(
        None,
        x=probe,
        cost_function=cost_function,
        grad=grad,
        num_iter=num_iter,
    )

    logger.info('%10s cost is %+12.5e', 'probe', cost)
    return probe, cost


def update_object(op, nearplane, probe, scan, psi, μ, τ, num_iter=1):
    """Solve the nearplane object recovery problem."""

    def cost_function(psi):
        return τ * np.linalg.norm(
            op.diffraction.fwd(psi=psi, scan=scan, probe=probe) - nearplane +
            μ / τ)**2

    def grad(psi):
        return τ * op.diffraction.adj(
            nearplane=(op.diffraction.fwd(psi=psi, scan=scan, probe=probe) -
                       nearplane + μ / τ),
            scan=scan,
            probe=probe,
        )

    psi, cost = conjugate_gradient(
        None,
        x=psi,
        cost_function=cost_function,
        grad=grad,
        num_iter=num_iter,
    )

    logger.info('%10s cost is %+12.5e', 'object', cost)
    return psi, cost
