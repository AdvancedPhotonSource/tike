import logging

import cupy as cp

from tike.linalg import lstsq, projection, norm, orthogonalize_gs
from tike.opt import batch_indicies, get_batch, put_batch

from ..position import update_positions_pd
from ..probe import orthogonalize_eig, get_varying_probe, update_eigen_probe

logger = logging.getLogger(__name__)


def lstsq_grad(
    op, comm,
    data, probe, scan, psi,
    recover_psi=True, recover_probe=False, recover_positions=False,
    cg_iter=4,
    cost=None,
    eigen_probe=None,
    eigen_weights=None,
    num_batch=1,
    subset_is_random=True,
    probe_is_orthogonal=False,
):  # yapf: disable
    """Solve the ptychography problem using Odstrcil et al's approach.

    The near- and farfield- ptychography problems are solved separately using
    gradient descent in the farfield and linear-least-squares in the nearfield.

    Parameters
    ----------
    op : tike.operators.Ptycho
        A ptychography operator.
    comm : tike.communicators.Comm
        An object which manages communications between GPUs and nodes.

    References
    ----------
    Michal Odstrcil, Andreas Menzel, and Manuel Guizar-Sicaros. Iterative
    least-squares solver for generalized maximum-likelihood ptychography.
    Optics Express. 2018.

    """
    # Unique batch for each device
    batches = [
        batch_indicies(s.shape[-2], num_batch, subset_is_random) for s in scan
    ]

    for n in range(num_batch):

        bdata = comm.pool.map(get_batch, data, batches, n=n)
        bscan = comm.pool.map(get_batch, scan, batches, n=n)

        if isinstance(eigen_probe, list):
            beigen_weights = comm.pool.map(
                get_batch,
                eigen_weights,
                batches,
                n=n,
            )
            beigen_probe = eigen_probe
        else:
            beigen_probe = [None] * comm.pool.num_workers
            beigen_weights = [None] * comm.pool.num_workers

        unique_probe = comm.pool.map(
            get_varying_probe,
            probe,
            beigen_probe,
            beigen_weights,
        )

        nearplane, cost = zip(*comm.pool.map(
            _update_wavefront,
            bdata,
            unique_probe,
            bscan,
            psi,
            op=op,
        ))

        if comm.use_mpi:
            cost = comm.Allreduce_reduce(cost, 'cpu')
        else:
            cost = comm.reduce(cost, 'cpu')

        (
            psi,
            probe,
            beigen_probe,
            beigen_weights,
        ) = _update_nearplane(
            op,
            comm,
            nearplane,
            psi,
            bscan,
            probe,
            unique_probe,
            beigen_probe,
            beigen_weights,
            recover_psi,
            recover_probe,
            probe_is_orthogonal,
        )

        if isinstance(eigen_probe, list):
            comm.pool.map(
                put_batch,
                beigen_weights,
                eigen_weights,
                batches,
                n=n,
            )

    result = {
        'psi': psi,
        'probe': probe,
        'cost': cost,
        'scan': scan,
    }
    if isinstance(eigen_probe, list):
        result['eigen_probe'] = eigen_probe
        result['eigen_weights'] = eigen_weights

    return result


def _get_nearplane_gradients(nearplane, psi, scan_, probe, unique_probe,
                             op, m, recover_psi, recover_probe):

    pad, end = op.diffraction.pad, op.diffraction.end

    patches = op.diffraction.patch.fwd(
        patches=cp.zeros(nearplane[..., m:m + 1, pad:end, pad:end].shape,
                         dtype='complex64')[..., 0, 0, :, :],
        images=psi,
        positions=scan_,
    )[..., None, None, :, :]

    # χ (diff) is the target for the nearplane problem; the difference
    # between the desired nearplane and the current nearplane that we wish
    # to minimize.
    diff = nearplane[..., m:m + 1, pad:end,
                     pad:end] - unique_probe[..., m:m + 1, :, :] * patches

    logger.info('%10s cost is %+12.5e', 'nearplane', norm(diff))

    if recover_psi:
        grad_psi = cp.conj(unique_probe[..., m:m + 1, :, :]) * diff

        # (25b) Common object gradient. Use a weighted (normalized) sum
        # instead of division as described in publication to improve
        # numerical stability.
        common_grad_psi = op.diffraction.patch.adj(
            patches=grad_psi[..., 0, 0, :, :],
            images=cp.zeros(psi.shape, dtype='complex64'),
            positions=scan_,
        )

        dOP = op.diffraction.patch.fwd(
            patches=cp.zeros(patches.shape, dtype='complex64')[..., 0,
                                                               0, :, :],
            images=common_grad_psi,
            positions=scan_,
        )[..., None, None, :, :] * unique_probe[..., m:m + 1, :, :]
        A1 = cp.sum((dOP * dOP.conj()).real + 0.5, axis=(-2, -1))
        A1 += 0.5 * cp.mean(A1, axis=-3, keepdims=True)
        b1 = cp.sum((dOP.conj() * diff).real, axis=(-2, -1))

    if recover_probe:
        grad_probe = cp.conj(patches) * diff

        # (25a) Common probe gradient. Use simple average instead of
        # division as described in publication because that's what
        # ptychoshelves does
        common_grad_probe = cp.mean(
            grad_probe,
            axis=-5,
            keepdims=True,
        )

        dPO = common_grad_probe * patches
        A4 = cp.sum((dPO * dPO.conj()).real + 0.5, axis=(-2, -1))
        A4 += 0.5 * cp.mean(A4, axis=-3, keepdims=True)
        b2 = cp.sum((dPO.conj() * diff).real, axis=(-2, -1))

    # (22) Use least-squares to find the optimal step sizes simultaneously
    if recover_psi and recover_probe:
        A2 = cp.sum((dOP * dPO.conj()), axis=(-2, -1))
        A3 = A2.conj()
        determinant = A1 * A4 - A2 * A3
        x1 = -cp.conj(A2 * b2 - A4 * b1) / determinant
        x2 = cp.conj(A1 * b2 - A3 * b1) / determinant
    elif recover_psi:
        x1 = b1 / A1
    elif recover_probe:
        x2 = b2 / A4

    weighted_step_psi = None
    weighted_step_probe = None

    if recover_psi:
        step = x1[..., None, None]

        # (27b) Object update
        weighted_step_psi = cp.mean(step, keepdims=True, axis=-5)

    if recover_probe:
        step = x2[..., None, None]

        weighted_step_probe = cp.mean(step, axis=-5, keepdims=True)

    if __debug__:
        patches = op.diffraction.patch.fwd(
            patches=cp.zeros(nearplane[..., m:m + 1, pad:end,
                                       pad:end].shape,
                             dtype='complex64')[..., 0, 0, :, :],
            images=psi,
            positions=scan_,
        )[..., None, None, :, :]
        logger.info(
            '%10s cost is %+12.5e', 'nearplane',
            norm(probe[..., m:m + 1, :, :] * patches -
                 nearplane[..., m:m + 1, pad:end, pad:end]))

    return (patches, diff, grad_probe, common_grad_psi, common_grad_probe,
            weighted_step_psi, weighted_step_probe)


def _get_residuals(grad_probe, grad_probe_mean):
    return grad_probe - grad_probe_mean


def _update_residuals(R, eigen_probe, axis, c, m):
    R -= projection(
        R,
        eigen_probe[..., c:c + 1, m:m + 1, :, :],
        axis=axis,
    )
    return R


def _update_nearplane(op, comm, nearplane, psi, scan_, probe, unique_probe,
                      eigen_probe, eigen_weights, recover_psi, recover_probe,
                      probe_is_orthogonal):

    for m in range(probe[0].shape[-3]):

        (
            patches,
            diff,
            grad_probe,
            common_grad_psi,
            common_grad_probe,
            weighted_step_psi,
            weighted_step_probe,
        ) = (list(a) for a in zip(*comm.pool.map(
            _get_nearplane_gradients,
            nearplane,
            psi,
            scan_,
            probe,
            unique_probe,
            op=op,
            m=m,
            recover_psi=recover_psi,
            recover_probe=recover_probe,
        )))

        if recover_probe and eigen_probe[0] is not None:
            logger.info('Updating eigen probes')
            # (30) residual probe updates
            grad_probe_mean = comm.pool.bcast(
                comm.pool.reduce_mean(
                    common_grad_probe,
                    axis=-5,
                ))
            R = comm.pool.map(_get_residuals, grad_probe, grad_probe_mean)

            for c in range(eigen_probe[0].shape[-4]):

                a, b = update_eigen_probe(
                    comm,
                    R,
                    [p[..., c:c + 1, m:m + 1, :, :] for p in eigen_probe],
                    [w[..., c, m] for w in eigen_weights],
                    patches,
                    diff,
                    β=0.01,  # TODO: Adjust according to mini-batch size
                )
                for p, w, x, y in zip(eigen_probe, eigen_weights, a, b):
                    p[..., c:c + 1, m:m + 1, :, :] = x
                    w[..., c, m] = y

                if eigen_probe[0].shape[-4] <= c + 1:
                    # Subtract projection of R onto new probe from R
                    R = comm.pool.map(
                        _update_residuals,
                        R,
                        eigen_probe,
                        axis=(-2, -1),
                        c=c,
                        m=m,
                    )

        # Update each direction
        if recover_psi:
            weighted_step_psi[0] = comm.pool.reduce_mean(
                weighted_step_psi,
                axis=-5,
            )[..., 0, 0, 0]
            common_grad_psi[0] = comm.pool.reduce_gpu(common_grad_psi)

            psi[0] += weighted_step_psi[0] * common_grad_psi[0]
            psi = comm.pool.bcast(psi[0])

        if recover_probe:
            weighted_step_probe[0] = comm.pool.reduce_mean(
                weighted_step_probe,
                axis=-5,
            )
            common_grad_probe[0] = comm.pool.reduce_mean(
                common_grad_probe,
                axis=-5,
            )

            # (27a) Probe update
            probe[0][..., m:m + 1, :, :] += (weighted_step_probe[0] *
                                             common_grad_probe[0])

        if probe[0].shape[-3] > 1 and probe_is_orthogonal:
            probe[0] = orthogonalize_gs(probe[0], axis=(-2, -1))
        probe = comm.pool.bcast(probe[0])

    return psi, probe, eigen_probe, eigen_weights


def _update_wavefront(data, varying_probe, scan, psi, op):

    # Compute the diffraction patterns for all of the probe modes at once.
    # We need access to all of the modes of a position to solve the phase
    # problem. The Ptycho operator doesn't do this natively, so it's messy.
    patches = cp.zeros(data.shape, dtype='complex64')
    patches = op.diffraction.patch.fwd(
        patches=patches,
        images=psi,
        positions=scan,
        patch_width=varying_probe.shape[-1],
    )
    patches = patches.reshape(op.ntheta, scan.shape[-2], 1, 1,
                              op.detector_shape, op.detector_shape)

    nearplane = cp.tile(patches, reps=(1, 1, 1, varying_probe.shape[-3], 1, 1))
    pad, end = op.diffraction.pad, op.diffraction.end
    nearplane[..., pad:end, pad:end] *= varying_probe

    # Solve the farplane phase problem ----------------------------------------
    farplane = op.propagation.fwd(nearplane, overwrite=True)
    intensity = cp.sum(cp.square(cp.abs(farplane)), axis=(2, 3))
    cost = op.propagation.cost(data, intensity)
    logger.info('%10s cost is %+12.5e', 'farplane', cost)
    farplane -= 0.5 * op.propagation.grad(data, farplane, intensity)

    if __debug__:
        intensity = cp.sum(cp.square(cp.abs(farplane)), axis=(2, 3))
        cost = op.propagation.cost(data, intensity)
        logger.info('%10s cost is %+12.5e', 'farplane', cost)
        # TODO: Only compute cost every 20 iterations or on a log sampling?

    farplane = op.propagation.adj(farplane, overwrite=True)

    return farplane, cost
