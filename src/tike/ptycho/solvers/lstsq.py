import logging

import cupy as cp

from tike.linalg import projection, norm, orthogonalize_gs
from tike.opt import randomizer, get_batch, put_batch, adam

from ..position import PositionOptions, update_positions_pd, _image_grad
from ..object import positivity_constraint, smoothness_constraint
from ..probe import (orthogonalize_eig, get_varying_probe, update_eigen_probe,
                     constrain_variable_probe)

logger = logging.getLogger(__name__)


def lstsq_grad(
    op,
    comm,
    data,
    batches,
    *,
    probe,
    scan,
    psi,
    algorithm_options,
    eigen_probe=None,
    eigen_weights=None,
    probe_options=None,
    position_options=None,
    object_options=None,
):
    """Solve the ptychography problem using Odstrcil et al's approach.

    Object and probe are updated simultaneouly using optimal step sizes
    computed using a least squares approach.

    Parameters
    ----------
    op : :py:class:`tike.operators.Ptycho`
        A ptychography operator.
    comm : :py:class:`tike.communicators.Comm`
        An object which manages communications between GPUs and nodes.
    data : list((FRAME, WIDE, HIGH) float32, ...)
        A list of unique CuPy arrays for each device containing
        the intensity (square of the absolute value) of the propagated
        wavefront; i.e. what the detector records. FFT-shifted so the
        diffraction peak is at the corners.
    batches : list(list((BATCH_SIZE, ) int, ...), ...)
        A list of list of indices along the FRAME axis of `data` for
        each device which define the batches of `data` to process
        simultaneously.
    probe : list((1, 1, SHARED, WIDE, HIGH) complex64, ...)
        A list of duplicate CuPy arrays for each device containing
        the shared complex illumination function amongst all positions.
    scan : list((POSI, 2) float32, ...)
        A list of unique CuPy arrays for each device containing
        coordinates of the minimum corner of the probe grid for each
        measurement in the coordinate system of psi. Coordinate order
        consistent with WIDE, HIGH order.
    psi : list((WIDE, HIGH) complex64, ...)
        A list of duplicate CuPy arrays for each device containing
        the wavefront modulation coefficients of the object.
    algorithm_options : :py:class:`tike.ptycho.IterativeOptions`
        The options class for this algorithm.
    position_options : :py:class:`tike.ptycho.PositionOptions`
        A class containing settings related to position correction.
    probe_options : :py:class:`tike.ptycho.ProbeOptions`
        A class containing settings related to probe updates.
    object_options : :py:class:`tike.ptycho.ObjectOptions`
        A class containing settings related to object updates.

    Returns
    -------
    result : dict
        A dictionary containing the updated keyword-only arguments passed to
        this function.

    References
    ----------
    Michal Odstrcil, Andreas Menzel, and Manuel Guizar-Sicaros. Iterative
    least-squares solver for generalized maximum-likelihood ptychography.
    Optics Express. 2018.

    .. seealso:: :py:mod:`tike.ptycho`

    """
    total_illumination = comm.pool.map(
        _get_total_illumination,
        psi,
        probe,
        scan,
        op=op,
    )

    total_patches = comm.pool.map(
        _get_total_patches,
        psi,
        probe,
        scan,
        op=op,
    )

    for n in randomizer.permutation(len(batches[0])):

        bdata = comm.pool.map(get_batch, data, batches, n=n)
        bscan = comm.pool.map(get_batch, scan, batches, n=n)

        if position_options:
            bposition_options = comm.pool.map(PositionOptions.split,
                                              position_options,
                                              [b[n] for b in batches])
        else:
            bposition_options = None

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
            bscan,
            bposition_options,
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
            object_options is not None,
            probe_options is not None,
            bposition_options,
            total_illumination,
            total_patches,
        )

        if position_options:
            comm.pool.map(PositionOptions.join, position_options,
                          bposition_options, [b[n] for b in batches])

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

    if probe_options and probe_options.orthogonality_constraint:
        probe[0] = orthogonalize_gs(probe[0], axis=(-2, -1))
        probe = comm.pool.bcast([probe[0]])

    if object_options:
        psi = comm.pool.map(positivity_constraint,
                            psi,
                            r=object_options.positivity_constraint)

        psi = comm.pool.map(smoothness_constraint,
                            psi,
                            a=object_options.smoothness_constraint)

    if isinstance(eigen_probe, list):
        eigen_probe, eigen_weights = (list(a) for a in zip(*comm.pool.map(
            constrain_variable_probe,
            eigen_probe,
            eigen_weights,
        )))

    algorithm_options.costs.append(cost)
    return {
        'probe': probe,
        'psi': psi,
        'scan': scan,
        'eigen_probe': eigen_probe,
        'eigen_weights': eigen_weights,
        'algorithm_options': algorithm_options,
        'probe_options': probe_options,
        'object_options': object_options,
        'position_options': position_options,
    }


def _get_total_illumination(psi, probe, scan, op=None):
    """Return the illumination of the primary probe."""
    _total_illumination = list()
    for i in range(probe.shape[-3]):
        primary = probe[0, 0, i, :, :]
        primary = primary * primary.conj()
        primary = cp.copy(
            cp.broadcast_to(primary, (*scan.shape[:-1], *probe.shape[-2:])))
        total_illumination = op.diffraction.patch.adj(
            patches=primary,
            images=cp.ones(psi.shape, dtype='complex64') * 1e-8j,
            positions=scan,
        ).real / 2
        total_illumination = cp.sqrt(total_illumination**2 +
                                     (0.1 * total_illumination.max())**2)
        assert cp.all(cp.isfinite(total_illumination))
        _total_illumination.append(total_illumination)
    return _total_illumination


def _get_total_patches(psi, probe, scan, op):
    patches = op.diffraction.patch.fwd(
        patches=cp.zeros(
            (*scan.shape[:-1], *probe.shape[-2:]),
            dtype='complex64',
        ),
        images=psi,
        positions=scan,
    )
    total_patches = cp.sum(patches * patches.conj(), axis=-3).real
    assert cp.all(cp.isfinite(total_patches))
    return total_patches + total_patches.max()


def _get_nearplane_gradients(
    nearplane,
    psi,
    scan_,
    probe,
    unique_probe,
    patches,
    total_illumination,
    total_patches,
    *,
    op,
    m,
    recover_psi,
    recover_probe,
):

    # χ (diff) is the target for the nearplane problem; the difference
    # between the desired nearplane and the current nearplane that we wish
    # to minimize.
    diff = nearplane[..., [m], :, :]
    assert cp.all(cp.isfinite(diff))

    logger.info('%10s cost is %+12.5e', 'nearplane', norm(diff))

    if recover_psi:
        grad_psi = cp.conj(unique_probe[..., [m], :, :]) * diff  # (24b)

        # (25b) Common object gradient. Use a weighted (normalized) sum
        # instead of division as described in publication to improve
        # numerical stability.
        common_grad_psi = op.diffraction.patch.adj(
            patches=grad_psi[..., 0, 0, :, :],
            images=cp.zeros(psi.shape, dtype='complex64'),
            positions=scan_,
        ) / total_illumination[m]
        assert cp.all(cp.isfinite(common_grad_psi))

        dOP = grad_psi * unique_probe[..., [m], :, :]
        assert cp.all(cp.isfinite(dOP))
        A1 = cp.sum((dOP * dOP.conj()).real, axis=(-2, -1)) + 0.5
    else:
        common_grad_psi = None
        dOP = None
        A1 = None

    if recover_probe:
        grad_probe = cp.conj(patches) * diff  # (24a)

        # (25a) Common probe gradient. Use simple average instead of
        # division as described in publication because that's what
        # ptychoshelves does
        common_grad_probe = cp.sum(
            grad_probe,
            axis=-5,
            keepdims=True,
        ) / total_patches
        # common_grad_probe = cp.mean(
        #     grad_probe,
        #     axis=-5,
        #     keepdims=True,
        # )  # TODO: Must change multi-device behavior
        # assert cp.all(cp.isfinite(common_grad_probe))

        dPO = grad_probe * patches
        A4 = cp.sum((dPO * dPO.conj()).real + 0.5, axis=(-2, -1))
    else:
        grad_probe = None
        common_grad_probe = None
        dPO = None
        A4 = None

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
        assert cp.all(cp.isfinite(b1)), b1
        assert cp.all(cp.isfinite(A1)), A1
        x1 = b1 / A1
    elif recover_probe:
        b2 = cp.sum((dPO.conj() * diff).real, axis=(-2, -1))
        x2 = b2 / A4

    if recover_psi:
        step = x1[..., None, None]
        assert cp.all(cp.isfinite(step))
        num_weighted_step_psi = cp.sum(
            step * unique_probe[..., [m], :, :],
            keepdims=True,
            axis=-5,
        )
        den_weighted_step_psi = cp.sum(
            unique_probe[..., [m], :, :],
            keepdims=True,
            axis=-5,
        )
    else:
        num_weighted_step_psi = None
        den_weighted_step_psi = None

    if recover_probe:
        step = x2[..., None, None]
        assert cp.all(cp.isfinite(step))
        num_weighted_step_probe = cp.sum(
            step * patches,
            keepdims=True,
            axis=-5,
        )
        den_weighted_step_probe = cp.sum(
            patches,
            keepdims=True,
            axis=-5,
        )
    else:
        num_weighted_step_probe = None
        den_weighted_step_probe = None

    return (
        diff,
        grad_probe,
        common_grad_psi,
        common_grad_probe,
        num_weighted_step_psi,
        den_weighted_step_psi,
        num_weighted_step_probe,
        den_weighted_step_probe,
    )


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


def _get_patches(nearplane, psi, scan, op=None):
    patches = op.diffraction.patch.fwd(
        patches=cp.zeros(
            nearplane[..., 0, 0, :, :].shape,
            dtype='complex64',
        ),
        images=psi,
        positions=scan,
    )[..., None, None, :, :]
    return patches


def _update_nearplane(
    op,
    comm,
    nearplane,
    psi,
    scan_,
    probe,
    unique_probe,
    eigen_probe,
    eigen_weights,
    recover_psi,
    recover_probe,
    position_options,
    total_illumination,
    total_patches,
):

    patches = comm.pool.map(_get_patches, nearplane, psi, scan_, op=op)

    for m in range(probe[0].shape[-3]):

        (
            diff,
            grad_probe,
            common_grad_psi,
            common_grad_probe,
            num_weighted_step_psi,
            den_weighted_step_psi,
            num_weighted_step_probe,
            den_weighted_step_probe,
        ) = (list(a) for a in zip(*comm.pool.map(
            _get_nearplane_gradients,
            nearplane,
            psi,
            scan_,
            probe,
            unique_probe,
            patches,
            total_illumination,
            total_patches,
            op=op,
            m=m,
            recover_psi=recover_psi,
            recover_probe=recover_probe,
        )))

        # Update each direction
        if recover_psi:
            if comm.use_mpi:
                # weighted_step_psi[0] = comm.Allreduce_mean(
                #     weighted_step_psi,
                #     axis=-5,
                # )[..., 0, 0, 0]
                common_grad_psi = comm.Allreduce_reduce(
                    common_grad_psi,
                    dest='gpu',
                )[0]
            else:
                num_weighted_step_psi = comm.reduce(
                    num_weighted_step_psi,
                    'gpu',
                )[0][..., 0, 0, 0]
                den_weighted_step_psi = comm.reduce(
                    den_weighted_step_psi,
                    'gpu',
                )[0][..., 0, 0, 0]
                common_grad_psi = comm.reduce(
                    common_grad_psi,
                    'gpu',
                )[0]

            # (27b) Object update
            assert cp.all(cp.isfinite(common_grad_psi))
            assert cp.all(cp.isfinite(num_weighted_step_psi))
            assert cp.all(cp.isfinite(den_weighted_step_psi))
            psi[0] -= (common_grad_psi * num_weighted_step_psi /
                       den_weighted_step_psi)
            assert cp.all(cp.isfinite(psi[0]))
            psi = comm.pool.bcast([psi[0]])

        if recover_probe:
            if comm.use_mpi:
                # weighted_step_probe[0] = comm.Allreduce_mean(
                #     weighted_step_probe,
                #     axis=-5,
                # )
                common_grad_probe[0] = comm.Allreduce_mean(
                    common_grad_probe,
                    axis=-5,
                )
            else:
                num_weighted_step_probe = comm.reduce(
                    num_weighted_step_probe,
                    'gpu',
                )[0]
                den_weighted_step_probe = comm.reduce(
                    den_weighted_step_probe,
                    'gpu',
                )[0]
                common_grad_probe = comm.reduce(
                    common_grad_probe,
                    'gpu',
                )[0]

            # (27a) Probe update
            probe[0][..., [m], :, :] -= (common_grad_probe *
                                         num_weighted_step_probe /
                                         den_weighted_step_probe)
            probe = comm.pool.bcast([probe[0]])

        if position_options and m == 0:
            scan_, position_options = zip(*comm.pool.map(
                _update_position,
                position_options,
                diff,
                patches,
                scan_,
                unique_probe,
                m=m,
                op=op,
            ))

    return psi, probe, eigen_probe, eigen_weights, scan_, position_options


def _update_wavefront(data, varying_probe, scan, psi, op):

    intensity, farplane = op._compute_intensity(
        data,
        psi,
        scan,
        varying_probe,
    )
    cost = op.propagation.cost(data, intensity)
    logger.info('%10s cost is %+12.5e', 'farplane', cost)

    farplane = op.propagation.adj(
        farplane=0.1 * op.propagation.grad(
            data,
            farplane,
            intensity,
            overwrite=True,
        ),
        overwrite=True,
    )

    pad, end = op.diffraction.pad, op.diffraction.end
    return farplane[..., pad:end, pad:end], cost


def _mad(x, **kwargs):
    """Return the mean absolute deviation around the median."""
    return cp.mean(cp.abs(x - cp.median(x, **kwargs)), **kwargs)


def _update_position(
    position_options,
    diff,
    patches,
    scan,
    unique_probe,
    m,
    op,
):
    main_probe = unique_probe[..., m:m + 1, :, :]

    # According to the manuscript, we can either shift the probe or the object
    # and they are equivalent (in theory). Here we shift the object because
    # that is what ptychoshelves does.
    grad_x, grad_y = _image_grad(op, patches)

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
    if position_options.use_adaptive_moment:
        logger.info(
            "position correction with ADAptive Momemtum acceleration enabled.")
        step_x, position_options.vx, position_options.mx = adam(
            step_x,
            position_options.vx,
            position_options.mx,
            vdecay=position_options.vdecay,
            mdecay=position_options.mdecay)
        step_y, position_options.vy, position_options.my = adam(
            step_y,
            position_options.vy,
            position_options.my,
            vdecay=position_options.vdecay,
            mdecay=position_options.mdecay)

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
    logger.info('position update norm is %+.3e', norm(step_x))

    # print(cp.abs(step_x) > 0.5)
    # print(cp.abs(step_y) > 0.5)

    scan[..., 0] -= step_y
    scan[..., 1] -= step_x

    return scan, position_options
