import logging

import cupy as cp

from tike.opt import batch_indicies
from ..position import update_positions_pd
from ..probe import orthogonalize_eig

logger = logging.getLogger(__name__)


def lstsq_grad(
    op, pool,
    data, probe, scan, psi,
    recover_psi=True, recover_probe=False, recover_positions=False,
    cg_iter=4,
    batch_size=None,
    cost=None,
    subset_is_random=True,
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
    probe = probe[0]
    psi = psi[0]

    # Divide the scan positions into smaller batches to be processed
    # sequentially. Otherwise we run out memory processing all of
    # the diffraction patterns at the same time.
    for index in batch_indicies(data[0].shape[1], batch_size, subset_is_random):
        data_ = data[0][:, index]
        scan_ = scan[0][:, index]

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
        patches = patches.reshape(op.ntheta, scan_.shape[-2] // op.fly, op.fly,
                                  1, op.detector_shape, op.detector_shape)

        nearplane = op.xp.tile(patches, reps=(1, 1, 1, probe.shape[-3], 1, 1))
        pad, end = op.diffraction.pad, op.diffraction.end
        nearplane[..., pad:end, pad:end] *= probe

        # Solve the farplane phase problem
        farplane = op.propagation.fwd(nearplane, overwrite=False)
        intensity = xp.sum(xp.square(xp.abs(farplane)), axis=(2, 3))
        cost = op.propagation.cost(data_, intensity)
        farplane -= op.propagation.grad(data_, farplane, intensity)

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

                # FIXME: What to do when elements of this norm are zero?
                norm_probe = cp.ones_like(psi)
                dir_psi = cp.zeros_like(psi)

                # FIXME: Shape changes required for fly scans.
                intensity = cp.ones(
                    (*scan_.shape[:2], 1, 1, 1, 1),
                    dtype='complex64',
                ) * cp.square(cp.abs(probe_))
                norm_probe = op.diffraction._patch(
                    patches=intensity,
                    psi=norm_probe,
                    scan=scan_,
                    fwd=False,
                )
                dir_psi = op.diffraction._patch(
                    patches=grad_psi,
                    psi=dir_psi,
                    scan=scan_,
                    fwd=False,
                )

                dir_psi /= norm_probe

                dOP = cp.zeros((*scan_.shape[:2], *data_.shape[-2:]),
                               dtype='complex64')
                dOP = op.diffraction._patch(
                    patches=dOP,
                    psi=dir_psi,
                    scan=scan_,
                    fwd=True,
                )
                dOP = dOP.reshape(op.ntheta, scan_.shape[-2] // op.fly, op.fly,
                                  1, op.detector_shape, op.detector_shape)
                dOP[..., pad:end, pad:end] *= probe_

                updates.append(dOP.view('float32').reshape(lstsq_shape))

            if recover_probe:
                grad_probe = (chi_ * xp.conj(patches))[..., pad:end, pad:end]
                dir_probe = cp.sum(
                    grad_probe,
                    axis=(1, 2),
                    keepdims=True,
                ) / cp.sum(
                    cp.square(cp.abs(patches[..., pad:end, pad:end])),
                    axis=(1, 2),
                )

                dPO = patches.copy()
                dPO[..., pad:end, pad:end] *= dir_probe

                updates.append(dPO.view('float32').reshape(lstsq_shape))

            # Use least-squares to find the optimal step sizes simultaneously
            # for all search directions.
            if updates:
                A = cp.stack(updates, axis=-1)
                b = chi_.view('float32').reshape(lstsq_shape)
                steps = _lstsq(A, b)
            num_steps = 0
            d = 0

            # Update each direction
            if recover_psi:
                step = steps[..., num_steps, None, None]
                # logger.info('%10s step is %+12.5e', 'object', step)
                num_steps += 1

                weighted_step = cp.zeros_like(psi)
                # FIXME: Shape changes required for fly scans.
                intensity = cp.ones(
                    (*scan_.shape[:2], 1, 1, 1, 1),
                    dtype='complex64',
                ) * cp.square(cp.abs(probe_))
                weighted_step = op.diffraction._patch(
                    patches=step * intensity,
                    psi=weighted_step,
                    scan=scan_,
                    fwd=False,
                )

                psi += dir_psi * weighted_step / norm_probe
                d += step * dOP

            if recover_probe:
                step = steps[..., num_steps, None, None]
                num_steps += 1

                weighted_step = cp.sum(
                    step * cp.square(cp.abs(patches[..., pad:end, pad:end])),
                    axis=(1, 2),
                )

                # FIXME: What to do when elements of this norm are zero?
                norm_psi = cp.sum(
                    cp.square(cp.abs(patches[..., pad:end, pad:end])),
                    axis=(1, 2),
                ) + 1

                probe_ += dir_probe * weighted_step / norm_psi
                d += step * dPO

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


def _lstsq(a, b):
    """Return the least-squares solution for a @ x = b.

    This implementation, unlike cp.linalg.lstsq, allows a stack of matricies to
    be processed simultaneously. The input sizes of the matricies are as
    follows:
        a (..., M, N)
        b (..., M)
        x (...,    N)

    ...seealso:: https://github.com/numpy/numpy/issues/8720
                 https://github.com/cupy/cupy/issues/3062
    """
    # TODO: Using 'out' parameter of cp.matmul() may reduce memory footprint
    assert a.shape[:-1] == b.shape, (f"Leading dims of a {a.shape}"
                                     f"and b {b.shape} must be same!")
    aT = a.swapaxes(-2, -1)
    x = cp.linalg.inv(aT @ a) @ aT @ b[..., None]
    return x[..., 0]


if __name__ == "__main__":
    N = (3, 4)

    a = cp.random.rand(*N, 5, 2) + 1j * cp.random.rand(*N, 5, 2)
    b = cp.random.rand(*N, 5) + 1j * cp.random.rand(*N, 5)

    x = _lstsq(a.astype('complex64'), b.astype('complex64'))

    assert x.shape == (*N, 2)
