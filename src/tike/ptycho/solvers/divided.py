import logging

import numpy as np

from tike.opt import conjugate_gradient, line_search
from ..position import update_positions_pd

logger = logging.getLogger(__name__)


def divided(
    op,
    data, probe, scan, psi,
    recover_psi=True, recover_probe=True, recover_positions=False,
    cg_iter=4,
    **kwargs
):  # yapf: disable
    """Solve near- and farfield- ptychography problems separately.

    References
    ----------
    Michal Odstrcil, Andreas Menzel, and Manuel Guizar-Sicaros. Iteraive
    least-squares solver for generalized maximum-likelihood ptychography.
    Optics Express. 2018.

    """
    for m in range(probe.shape[-3]):
        intensity = op._compute_intensity(
            data,
            psi,
            scan,
            probe[..., np.r_[0:m, m + 1:probe.shape[-3]], :, :],
        )
        farplane = op.fwd(psi=psi, scan=scan, probe=probe[..., m:m + 1, :, :])

        farplane, cost = update_phase(op,
                                      data,
                                      farplane,
                                      intensity,
                                      num_iter=cg_iter)
        nearplane = op.propagation.adj(farplane, overwrite=True)

        if recover_psi:
            psi, cost = update_object(op,
                                    nearplane,
                                    probe[..., m:m + 1, :, :],
                                    scan,
                                    psi,
                                    num_iter=cg_iter)

        if recover_probe:
            probe, cost = update_probe(op,
                                    nearplane,
                                    probe[..., m:m + 1, :, :],
                                    scan,
                                    psi,
                                    num_iter=cg_iter)

    if recover_positions:
        scan, cost = update_positions_pd(op, data, psi, probe, scan)

    return {
        'psi': psi,
        'probe': probe,
        'cost': cost,
        'scan': scan,
        'farplane': farplane,
    }


def update_phase(op, data, farplane, intensity, num_iter=1):
    """Solve the farplane phase problem."""

    def grad(farplane):
        return op.propagation.grad(
            data,
            farplane,
            intensity + np.sum(
                np.square(np.abs(farplane)).reshape(
                    *data.shape[:2],
                    -1,
                    *data.shape[2:],
                ),
                axis=2,
            ),
        )

    def cost_function(farplane):
        return op.propagation.cost(
            data,
            intensity + np.sum(
                np.square(np.abs(farplane)).reshape(
                    *data.shape[:2],
                    -1,
                    *data.shape[2:],
                ),
                axis=2,
            ),
        )

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


def update_probe(op, nearplane, probe, scan, psi, num_iter=1):
    """Solve the nearplane single probe recovery problem."""
    xp = op.xp

    def cost_function(probe):
        return xp.linalg.norm(
            xp.ravel(
                op.diffraction.fwd(psi=psi, scan=scan, probe=probe) -
                nearplane))**2

    def grad(probe):
        # Use the average gradient for all probe positions
        return xp.mean(
            op.diffraction.adj_probe(
                op.diffraction.fwd(psi=psi, scan=scan, probe=probe) - nearplane,
                scan=scan,
                psi=psi,
            ),
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


def update_object(op, nearplane, probe, scan, psi, num_iter=1):
    """Solve the nearplane object recovery problem."""
    xp = op.xp

    def cost_function(psi):
        return xp.linalg.norm(
            xp.ravel(
                op.diffraction.fwd(psi=psi, scan=scan, probe=probe) -
                nearplane))**2

    def grad(psi):
        return op.diffraction.adj(
            nearplane=(op.diffraction.fwd(psi=psi, scan=scan, probe=probe) -
                       nearplane),
            scan=scan,
            probe=probe,
            overwrite=True,
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
