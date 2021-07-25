import logging

import cupy as cp
import cupyx.scipy.ndimage

from tike.linalg import lstsq, projection, norm, orthogonalize_gs
from tike.opt import batch_indicies, get_batch, put_batch, adam

from ..position import update_positions_pd, _image_grad
from ..probe import (orthogonalize_eig, get_varying_probe, update_eigen_probe,
                     constrain_variable_probe)

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
    positivity_constraint=0,
    smoothness_constraint=0,
    position_momentum=False,
    vx=None, vy=None, mx=None, my=None,
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
    if recover_positions and position_momentum and vx is None:
        vx = [cp.zeros(scan[0].shape[:2], dtype='float32')]
        vy = [cp.zeros(scan[0].shape[:2], dtype='float32')]
        mx = [cp.zeros(scan[0].shape[:2], dtype='float32')]
        my = [cp.zeros(scan[0].shape[:2], dtype='float32')]

    # Unique batch for each device
    batches = [
        batch_indicies(s.shape[-2], num_batch, subset_is_random) for s in scan
    ]

    for n in range(num_batch):

        bdata = comm.pool.map(get_batch, data, batches, n=n)
        bscan = comm.pool.map(get_batch, scan, batches, n=n)

        if vx is not None:
            bvx = vx[0][:, batches[0][n]]
            bvy = vy[0][:, batches[0][n]]
            bmx = mx[0][:, batches[0][n]]
            bmy = my[0][:, batches[0][n]]
        else:
            bvx = None
            bvy = None
            bmx = None
            bmy = None

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

        (psi, probe, beigen_probe, beigen_weights, bscan, bvx, bvy, bmx,
         bmy) = _update_nearplane(
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
             recover_positions,
             probe_is_orthogonal,
             bvx,
             bvy,
             bmx,
             bmy,
         )

        if vx is not None:
            vx[0][:, batches[0][n]] = bvx
            vy[0][:, batches[0][n]] = bvy
            mx[0][:, batches[0][n]] = bmx
            my[0][:, batches[0][n]] = bmy

        if isinstance(eigen_probe, list):
            comm.pool.map(
                put_batch,
                beigen_weights,
                eigen_weights,
                batches,
                n=n,
            )

        comm.pool.map(
            put_batch,
            bscan,
            scan,
            batches,
            n=n,
        )

    if probe[0].shape[-3] > 1 and probe_is_orthogonal:
        probe[0] = orthogonalize_gs(probe[0], axis=(-2, -1))
        probe = comm.pool.bcast([probe[0]])

    psi = comm.pool.map(_positivity_constraint, psi, r=positivity_constraint)

    psi = comm.pool.map(_smoothness_constraint, psi, a=smoothness_constraint)

    if isinstance(eigen_probe, list):
        eigen_probe, eigen_weights = (list(a) for a in zip(*comm.pool.map(
            constrain_variable_probe,
            eigen_probe,
            eigen_weights,
        )))

    result = {
        'psi': psi,
        'probe': probe,
        'cost': cost,
        'scan': scan,
    }
    if isinstance(eigen_probe, list):
        result['eigen_probe'] = eigen_probe
        result['eigen_weights'] = eigen_weights
    if vx is not None:
        result['vx'] = vx
        result['vy'] = vy
        result['mx'] = mx
        result['my'] = my

    return result


def _positivity_constraint(x, r):
    if r > 0:
        return r * cp.abs(x) + (1 - r) * x
    else:
        return x


def _smoothness_constraint(x, a):
    """Convolves the image with a 3x3 averaging kernel.

    The kernel is defined as

    [[a, a, a]
     [a, c, a]
     [a, a, a]]

    where c = 1 - 8 * a

    Parameters
    ----------
    a : float [0, 1/8)
        The non-center weights of the kernel.
    """
    if 0 <= a and a < 1.0 / 8.0:
        weights = cp.ones([1] * (x.ndim - 2) + [3, 3], dtype='float32') * a
        weights[..., 1, 1] = 1.0 - 8.0 * a
        x.real = cupyx.scipy.ndimage.convolve(x.real, weights, mode='nearest')
        x.imag = cupyx.scipy.ndimage.convolve(x.imag, weights, mode='nearest')
    else:
        raise ValueError(
            f"Smoothness constraint must be in range [0, 1/8) not {a}.")
    return x


def _get_nearplane_gradients(nearplane, psi, scan_, probe, unique_probe, op, m,
                             recover_psi, recover_probe, recover_positions):

    pad, end = op.diffraction.pad, op.diffraction.end

    patches = op.diffraction.patch.fwd(
        patches=cp.zeros(nearplane[..., [m], pad:end, pad:end].shape,
                         dtype='complex64')[..., 0, 0, :, :],
        images=psi,
        positions=scan_,
    )[..., None, None, :, :]

    # χ (diff) is the target for the nearplane problem; the difference
    # between the desired nearplane and the current nearplane that we wish
    # to minimize.
    diff = nearplane[..., [m], pad:end,
                     pad:end] - unique_probe[..., [m], :, :] * patches

    logger.info('%10s cost is %+12.5e', 'nearplane', norm(diff))

    if recover_psi:
        grad_psi = cp.conj(unique_probe[..., [m], :, :]) * diff

        # (25b) Common object gradient. Use a weighted (normalized) sum
        # instead of division as described in publication to improve
        # numerical stability.
        common_grad_psi = op.diffraction.patch.adj(
            patches=grad_psi[..., 0, 0, :, :],
            images=cp.zeros(psi.shape, dtype='complex64'),
            positions=scan_,
        )

        dOP = op.diffraction.patch.fwd(
            patches=cp.zeros(patches.shape, dtype='complex64')[..., 0, 0, :, :],
            images=common_grad_psi,
            positions=scan_,
        )[..., None, None, :, :] * unique_probe[..., [m], :, :]
        A1 = cp.sum((dOP * dOP.conj()).real + 0.5, axis=(-2, -1))
    else:
        common_grad_psi = None
        dOP = None
        A1 = None

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
    else:
        grad_probe = None
        common_grad_probe = None
        dPO = None
        A4 = None

    if __debug__:
        patches = op.diffraction.patch.fwd(
            patches=cp.zeros(nearplane[..., [m], pad:end, pad:end].shape,
                             dtype='complex64')[..., 0, 0, :, :],
            images=psi,
            positions=scan_,
        )[..., None, None, :, :]
        logger.info(
            '%10s cost is %+12.5e', 'nearplane',
            norm(probe[..., [m], :, :] * patches -
                 nearplane[..., [m], pad:end, pad:end]))

    return (patches, diff, grad_probe, common_grad_psi, common_grad_probe, dOP,
            dPO, A1, A4)


def _get_nearplane_steps(diff, dOP, dPO, A1, A4, recover_psi, recover_probe):
    # (22) Use least-squares to find the optimal step sizes simultaneously
    if recover_psi and recover_probe:
        b1 = cp.sum((dOP.conj() * diff).real, axis=(-2, -1))
        b2 = cp.sum((dPO.conj() * diff).real, axis=(-2, -1))
        A2 = cp.sum((dOP * dPO.conj()), axis=(-2, -1))
        A3 = A2.conj()
        determinant = A1 * A4 - A2 * A3
        x1 = -cp.conj(A2 * b2 - A4 * b1) / determinant
        x2 = cp.conj(A1 * b2 - A3 * b1) / determinant
    elif recover_psi:
        b1 = cp.sum((dOP.conj() * diff).real, axis=(-2, -1))
        x1 = b1 / A1
    elif recover_probe:
        b2 = cp.sum((dPO.conj() * diff).real, axis=(-2, -1))
        x2 = b2 / A4

    if recover_psi:
        step = x1[..., None, None]

        # (27b) Object update
        weighted_step_psi = cp.mean(step, keepdims=True, axis=-5)

    if recover_probe:
        step = x2[..., None, None]

        weighted_step_probe = cp.mean(step, axis=-5, keepdims=True)
    else:
        weighted_step_probe = None

    return weighted_step_psi, weighted_step_probe


def _update_A(A, delta):
    A += 0.5 * delta
    return A


def _get_residuals(grad_probe, grad_probe_mean):
    return grad_probe - grad_probe_mean


def _update_residuals(R, eigen_probe, axis, c, m):
    R -= projection(
        R,
        eigen_probe[..., c:c + 1, m:m + 1, :, :],
        axis=axis,
    )
    return R


def _get_coefs_intensity(weights, xi, P, O, m):
    OP = O * P
    num = cp.sum(cp.real(cp.conj(OP) * xi), axis=(-1, -2))
    den = cp.sum(cp.abs(OP)**2, axis=(-1, -2))
    weights[..., 0:1, m:m + 1] = weights[..., 0:1, m:m + 1] + 0.1 * num / den
    return weights


def _update_nearplane(op, comm, nearplane, psi, scan_, probe, unique_probe,
                      eigen_probe, eigen_weights, recover_psi, recover_probe,
                      recover_positions, probe_is_orthogonal, vx, vy, mx, my):

    for m in range(probe[0].shape[-3]):

        (
            patches,
            diff,
            grad_probe,
            common_grad_psi,
            common_grad_probe,
            dOP,
            dPO,
            A1,
            A4,
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
            recover_positions=recover_positions,
        )))

        if recover_psi:
            if comm.use_mpi:
                delta = comm.Allreduce_mean(A1, axis=-3)
            else:
                delta = comm.pool.reduce_mean(A1, axis=-3)
            A1 = comm.pool.map(_update_A, A1, comm.pool.bcast([delta]))

        if recover_probe:
            if comm.use_mpi:
                delta = comm.Allreduce_mean(A4, axis=-3)
            else:
                delta = comm.pool.reduce_mean(A4, axis=-3)
            A4 = comm.pool.map(_update_A, A4, comm.pool.bcast([delta]))

        if recover_probe or recover_psi:
            (
                weighted_step_psi,
                weighted_step_probe,
            ) = (list(a) for a in zip(*comm.pool.map(
                _get_nearplane_steps,
                diff,
                dOP,
                dPO,
                A1,
                A4,
                recover_psi=recover_psi,
                recover_probe=recover_probe,
            )))

        if recover_probe and eigen_weights[0] is not None:
            logger.info('Updating eigen probes')

            eigen_weights = comm.pool.map(
                _get_coefs_intensity,
                eigen_weights,
                diff,
                [p[..., m:m + 1, :, :] for p in probe],
                patches,
                m=m,
            )

            # (30) residual probe updates
            if eigen_weights[0].shape[-2] > 1:
                if comm.use_mpi:
                    grad_probe_mean = comm.Allreduce_mean(
                        common_grad_probe,
                        axis=-5,
                    )
                    grad_probe_mean = comm.pool.bcast([grad_probe_mean])
                else:
                    grad_probe_mean = comm.pool.bcast(
                        [comm.pool.reduce_mean(
                            common_grad_probe,
                            axis=-5,
                        )])
                R = comm.pool.map(_get_residuals, grad_probe, grad_probe_mean)

            for n in range(1, eigen_weights[0].shape[-2]):

                a, b = update_eigen_probe(
                    comm,
                    R,
                    [p[..., n - 1:n, m:m + 1, :, :] for p in eigen_probe],
                    [w[..., n, m] for w in eigen_weights],
                    patches,
                    diff,
                    β=0.01,  # TODO: Adjust according to mini-batch size
                )
                for p, w, x, y in zip(eigen_probe, eigen_weights, a, b):
                    p[..., n - 1:n, m:m + 1, :, :] = x
                    w[..., n, m] = y

                if n + 1 < eigen_weights[0].shape[-2]:
                    # Subtract projection of R onto new probe from R
                    R = comm.pool.map(
                        _update_residuals,
                        R,
                        eigen_probe,
                        axis=(-2, -1),
                        c=n - 1,
                        m=m,
                    )

        # Update each direction
        if recover_psi:
            if comm.use_mpi:
                weighted_step_psi[0] = comm.Allreduce_mean(
                    weighted_step_psi,
                    axis=-5,
                )[..., 0, 0, 0]
                common_grad_psi[0] = comm.Allreduce_reduce(
                    common_grad_psi,
                    dest='gpu',
                )[0]
            else:
                weighted_step_psi[0] = comm.pool.reduce_mean(
                    weighted_step_psi,
                    axis=-5,
                )[..., 0, 0, 0]
                common_grad_psi[0] = comm.reduce(common_grad_psi, 'gpu')[0]

            psi[0] += weighted_step_psi[0] * common_grad_psi[0]
            psi = comm.pool.bcast([psi[0]])

        if recover_probe:
            if comm.use_mpi:
                weighted_step_probe[0] = comm.Allreduce_mean(
                    weighted_step_probe,
                    axis=-5,
                )
                common_grad_probe[0] = comm.Allreduce_mean(
                    common_grad_probe,
                    axis=-5,
                )
            else:
                weighted_step_probe[0] = comm.pool.reduce_mean(
                    weighted_step_probe,
                    axis=-5,
                )
                common_grad_probe[0] = comm.pool.reduce_mean(
                    common_grad_probe,
                    axis=-5,
                )

            # (27a) Probe update
            probe[0][..., [m], :, :] += (weighted_step_probe[0] *
                                         common_grad_probe[0])
            probe = comm.pool.bcast([probe[0]])

        if recover_positions and m == 0:
            scan_[0], vx, vy, mx, my = _update_position(
                op,
                comm,
                diff[0],
                patches[0],
                scan_[0],
                unique_probe[0],
                m=m,
                vx=vx,
                vy=vy,
                mx=mx,
                my=my,
            )

    return psi, probe, eigen_probe, eigen_weights, scan_, vx, vy, mx, my


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
    patches = patches.reshape(*scan.shape[:-1], 1, 1, op.detector_shape,
                              op.detector_shape)

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


def _mad(x, **kwargs):
    """Return the mean absolute deviation around the median."""
    return cp.mean(cp.abs(x - cp.median(x, **kwargs)), **kwargs)


def _update_position(op,
                     comm,
                     diff,
                     patches,
                     scan,
                     unique_probe,
                     m,
                     vx=None,
                     vy=None,
                     mx=None,
                     my=None):

    main_probe = unique_probe[..., m:m + 1, :, :]

    # According to the manuscript, we can either shift the probe or the object
    # and they are equivalent (in theory). Here we shift the object because
    # that is what ptychoshelves does.
    grad_x, grad_y = _image_grad(patches)

    numerator = cp.sum(cp.real(diff * cp.conj(grad_x * main_probe)),
                       axis=(-2, -1))
    denominator = cp.sum(cp.abs(grad_x * main_probe)**2, axis=(-2, -1))
    step_x = numerator / (denominator + 1e-6)

    numerator = cp.sum(cp.real(diff * cp.conj(grad_y * main_probe)),
                       axis=(-2, -1))
    denominator = cp.sum(cp.abs(grad_y * main_probe)**2, axis=(-2, -1))
    step_y = numerator / (denominator + 1e-6)

    step_x = step_x[..., 0, 0]
    step_y = step_y[..., 0, 0]

    # Momentum
    if vx is not None:
        logger.info(
            "position correction with ADAptive Momemtum acceleration enabled.")
        step_x, vx, mx = adam(step_x, vx, mx, vdecay=0.4, mdecay=0.4)
        step_y, vy, my = adam(step_y, vy, my, vdecay=0.4, mdecay=0.4)

    # Step limit for stability
    max_shift = patches.shape[-1] * 0.1
    _max_shift = cp.minimum(
        max_shift,
        _mad(
            cp.concatenate((step_x, step_y), axis=-1),
            axis=-1,
            keepdims=True,
        ),
    )
    step_x = cp.maximum(-_max_shift, cp.minimum(step_x, _max_shift))
    step_y = cp.maximum(-_max_shift, cp.minimum(step_y, _max_shift))

    # SYNCHRONIZE ME?
    # step -= comm.Allreduce_mean(step)
    # Ensure net movement is zero
    step_x -= cp.mean(step_x, axis=-1, keepdims=True)
    step_y -= cp.mean(step_y, axis=-1, keepdims=True)
    logger.info(f'position update norm is {norm(step_x)}')

    # print(cp.abs(step_x) > 0.5)
    # print(cp.abs(step_y) > 0.5)

    scan[..., 0] -= step_y
    scan[..., 1] -= step_x

    return scan, vx, vy, mx, my
