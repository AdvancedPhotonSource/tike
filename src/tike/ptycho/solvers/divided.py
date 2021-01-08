import logging

import cupy as cp

from tike.linalg import lstsq

from ..position import update_positions_pd
from ..probe import orthogonalize_eig

logger = logging.getLogger(__name__)


def lstsq_grad(
    op, comm, pool,
    data, probe, scan, psi,
    recover_psi=True, recover_probe=False, recover_positions=False,
    cg_iter=4,
    cost=None,
):  # yapf: disable
    """Solve the ptychography problem using Odstrcil et al's approach.

    The near- and farfield- ptychography problems are solved separately using
    gradient descent in the farfield and linear-least-squares in the nearfield.

    Parameters
    ----------
    op : tike.operators.Ptycho
        A ptychography operator.
    pool : tike.pool.ThreadPoolExecutor
        An object which manages communications between GPUs.

    References
    ----------
    Michal Odstrcil, Andreas Menzel, and Manuel Guizar-Sicaros. Iteraive
    least-squares solver for generalized maximum-likelihood ptychography.
    Optics Express. 2018.

    """
    xp = op.xp
    data_ = data[0]
    probe = probe[0]
    scan_ = scan[0]
    psi = psi[0]

    # Compute the diffraction patterns for all of the probe modes at once.
    # We need access to all of the modes of a position to solve the phase
    # problem. The Ptycho operator doesn't do this natively, so it's messy.
    patches = cp.zeros(data_.shape, dtype='complex64')
    patches = op.diffraction._patch(
        patches=patches,
        psi=psi,
        scan=scan_,
        fwd=True,
    )
    patches = patches.reshape(op.ntheta, scan_.shape[-2], 1, 1,
                              op.detector_shape, op.detector_shape)

    nearplane = op.xp.tile(patches, reps=(1, 1, 1, probe.shape[-3], 1, 1))
    pad, end = op.diffraction.pad, op.diffraction.end
    nearplane[..., pad:end, pad:end] *= probe

    # Solve the farplane phase problem
    farplane = op.propagation.fwd(nearplane, overwrite=False)
    intensity = xp.sum(xp.square(xp.abs(farplane)), axis=(2, 3))
    cost = op.propagation.cost(data_, intensity)
    logger.info('%10s cost is %+12.5e', 'farplane', cost)
    farplane -= 0.5 * op.propagation.grad(data_, farplane, intensity)

    if __debug__:
        intensity = xp.sum(xp.square(xp.abs(farplane)), axis=(2, 3))
        cost = op.propagation.cost(data_, intensity)
        logger.info('%10s cost is %+12.5e', 'farplane', cost)
        # TODO: Only compute cost every 20 iterations or on a log sampling?

    # Use χ (chi) to solve the nearplane problem. We use least-squares to
    # find the update of all the search directions: object, probe,
    # positions, etc that causes the nearplane wavefront to match the one
    # we just found by solving the phase problem.
    farplane = op.propagation.adj(farplane, overwrite=True)
    chi = [
        farplane[..., m:m + 1, :, :] - nearplane[..., m:m + 1, :, :]
        for m in range(probe.shape[-3])
    ]

    # To solve the least-squares optimal step problem we flatten the last
    # two dimensions of the nearplanes and convert from complex to float
    lstsq_shape = (*nearplane.shape[:-3], 1,
                   nearplane.shape[-2] * nearplane.shape[-1] * 2)

    for m in range(probe.shape[-3]):
        chi_ = chi[m]
        probe_ = probe[..., m:m + 1, :, :]

        logger.info('%10s cost is %+12.5e', 'nearplane',
                    cp.linalg.norm(cp.ravel(chi_)))

        updates = []

        if recover_psi:
            # FIXME: Implement conjugate gradient
            grad_psi = chi_.copy()
            grad_psi[..., pad:end, pad:end] *= cp.conj(probe_)

            probe_intensity = cp.ones(
                (*scan_.shape[:2], 1, 1, 1, 1),
                dtype='complex64',
            ) * cp.square(cp.abs(probe_))

            norm_probe = op.diffraction._patch(
                patches=probe_intensity,
                psi=cp.ones_like(psi),
                scan=scan_,
                fwd=False,
            ) + 1e-6

            # FIXME: What to do when elements of this norm are zero?
            dir_psi = op.diffraction._patch(
                patches=grad_psi,
                psi=cp.zeros_like(psi),
                scan=scan_,
                fwd=False,
            ) / norm_probe

            dOP = op.diffraction._patch(
                patches=cp.zeros((*scan_.shape[:2], *data_.shape[-2:]),
                                 dtype='complex64'),
                psi=dir_psi,
                scan=scan_,
                fwd=True,
            )
            dOP = dOP.reshape(op.ntheta, scan_.shape[-2], 1, 1,
                              op.detector_shape, op.detector_shape)
            dOP[..., pad:end, pad:end] *= probe_

            updates.append(dOP.view('float32').reshape(lstsq_shape))

        if recover_probe:
            patches = op.diffraction._patch(
                patches=cp.zeros(data_.shape, dtype='complex64'),
                psi=psi,
                scan=scan_,
                fwd=True,
            )
            patches = patches.reshape(op.ntheta, scan_.shape[-2], 1, 1,
                                      op.detector_shape, op.detector_shape)

            grad_probe = (chi_ * xp.conj(patches))[..., pad:end, pad:end]

            psi_intensity = cp.square(cp.abs(patches[..., pad:end, pad:end]))

            norm_psi = cp.sum(psi_intensity, axis=1, keepdims=True) + 1e-6

            dir_probe = cp.sum(grad_probe, axis=1, keepdims=True) / norm_psi

            dPO = patches.copy()
            dPO[..., pad:end, pad:end] *= dir_probe

            updates.append(dPO.view('float32').reshape(lstsq_shape))

        # Use least-squares to find the optimal step sizes simultaneously
        # for all search directions.
        if updates:
            A = cp.stack(updates, axis=-1)
            b = chi_.view('float32').reshape(lstsq_shape)
            steps = lstsq(A, b)
        num_steps = 0
        d = 0

        # Update each direction
        if recover_psi:
            step = steps[..., num_steps, None, None]
            num_steps += 1

            weighted_step = op.diffraction._patch(
                patches=step * probe_intensity,
                psi=cp.zeros_like(psi),
                scan=scan_,
                fwd=False,
            )

            psi += dir_psi * weighted_step / norm_probe
            d += step * dOP

        if recover_probe:
            step = steps[..., num_steps, None, None]
            num_steps += 1

            weighted_step = cp.sum(step * psi_intensity, axis=1, keepdims=True)

            probe_ += dir_probe * weighted_step / norm_psi
            d += step * dPO

        if __debug__:
            logger.info('%10s cost is %+12.5e', 'nearplane',
                        cp.linalg.norm(cp.ravel(chi_ - d)))

    if probe.shape[-3] > 1:
        probe = orthogonalize_eig(probe)

    return {
        'psi': [psi],
        'probe': [probe],
        'cost': cost,
        'scan': scan,
    }
