import logging

import cupy as cp

import tike.linalg
import tike.opt
import tike.ptycho.probe
import tike.ptycho.position

from ..position import PositionOptions
from ..object import positivity_constraint, smoothness_constraint

logger = logging.getLogger(__name__)


def lstsq_grad(
    op,
    comm,
    data,
    batches,
    *,
    parameters,
):
    """Solve the ptychography problem using Odstrcil et al's approach.

    Object and probe are updated simultaneously using optimal step sizes
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
    parameters : :py:class:`tike.ptycho.solvers.PtychoParameters`
        An object which contains reconstruction parameters.

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
    probe = parameters.probe
    scan = parameters.scan
    psi = parameters.psi
    algorithm_options = parameters.algorithm_options
    probe_options = parameters.probe_options
    position_options = parameters.position_options
    object_options = parameters.object_options
    eigen_probe = parameters.eigen_probe
    eigen_weights = parameters.eigen_weights

    if eigen_probe is None:
        beigen_probe = [None] * comm.pool.num_workers
    else:
        beigen_probe = eigen_probe

    if object_options is not None:
        preconditioner = [None] * comm.pool.num_workers
        for n in range(len(batches[0])):
            bscan = comm.pool.map(tike.opt.get_batch, scan, batches, n=n)
            preconditioner = comm.pool.map(
                _psi_preconditioner,
                preconditioner,
                probe,
                bscan,
                psi,
                op=op,
            )
        if comm.use_mpi:
            preconditioner = comm.Allreduce(preconditioner)
        else:
            preconditioner = comm.pool.allreduce(preconditioner)
        # Use a rolling average of this preconditioner and the previous
        # preconditioner
        if object_options.preconditioner is None:
            object_options.preconditioner = preconditioner
        else:
            object_options.preconditioner = comm.pool.map(
                cp.add,
                object_options.preconditioner,
                preconditioner,
            )
            object_options.preconditioner = comm.pool.map(
                cp.divide,
                object_options.preconditioner,
                [2] * comm.pool.num_workers,
            )

        if algorithm_options.batch_method == 'cluster_compact':
            object_options.combined_update = cp.zeros_like(psi[0])

    batch_cost = []
    for n in tike.opt.randomizer.permutation(len(batches[0])):

        bdata = comm.pool.map(tike.opt.get_batch, data, batches, n=n)
        bscan = comm.pool.map(tike.opt.get_batch, scan, batches, n=n)

        if position_options is None:
            bposition_options = None
        else:
            bposition_options = comm.pool.map(
                PositionOptions.split,
                position_options,
                [b[n] for b in batches],
            )

        if eigen_weights is None:
            unique_probe = probe
            beigen_weights = [None] * comm.pool.num_workers
        else:
            beigen_weights = comm.pool.map(
                tike.opt.get_batch,
                eigen_weights,
                batches,
                n=n,
            )
            unique_probe = comm.pool.map(
                tike.ptycho.probe.get_varying_probe,
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
            batch_cost.append(comm.Allreduce_reduce(cost, 'cpu'))
        else:
            batch_cost.append(comm.reduce(cost, 'cpu'))

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
            num_batch=algorithm_options.num_batch,
            psi_update_denominator=object_options.preconditioner,
            object_options=object_options,
            probe_options=probe_options,
            algorithm_options=algorithm_options,
        )

        if position_options:
            comm.pool.map(
                PositionOptions.insert,
                position_options,
                bposition_options,
                [b[n] for b in batches],
            )

        if eigen_weights is not None:
            comm.pool.map(
                tike.opt.put_batch,
                beigen_weights,
                eigen_weights,
                batches,
                n=n,
            )

        comm.pool.map(
            tike.opt.put_batch,
            bscan,
            scan,
            batches,
            n=n,
        )

    if object_options and algorithm_options.batch_method == 'cluster_compact':
        # (27b) Object update
        dpsi = object_options.combined_update
        if object_options.use_adaptive_moment:
            (
                dpsi,
                object_options.v,
                object_options.m,
            ) = tike.opt.momentum(
                g=dpsi,
                v=object_options.v,
                m=object_options.m,
                vdecay=object_options.vdecay,
                mdecay=object_options.mdecay,
            )
        psi[0] = psi[0] + dpsi
        psi = comm.pool.bcast([psi[0]])

    if probe_options and probe_options.orthogonality_constraint:
        probe = comm.pool.map(tike.ptycho.probe.orthogonalize_eig, probe)

    if object_options:
        psi = comm.pool.map(positivity_constraint,
                            psi,
                            r=object_options.positivity_constraint)

        psi = comm.pool.map(smoothness_constraint,
                            psi,
                            a=object_options.smoothness_constraint)

    if eigen_probe is not None:
        eigen_probe, eigen_weights = tike.ptycho.probe.constrain_variable_probe(
            comm,
            beigen_probe,
            eigen_weights,
        )

    algorithm_options.costs.append(batch_cost)
    parameters.probe = probe
    parameters.psi = psi
    parameters.scan = scan
    parameters.algorithm_options = algorithm_options
    parameters.probe_options = probe_options
    parameters.object_options = object_options
    parameters.position_options = position_options
    parameters.eigen_weights = eigen_weights
    parameters.eigen_probe = eigen_probe
    return parameters


def _psi_preconditioner(psi_update_denominator,
                        unique_probe,
                        scan_,
                        psi,
                        *,
                        op,
                        m=0):
    # Sum of the probe amplitude over field of view for preconditioning the
    # object update.
    probe_amp = (unique_probe[..., 0, m, :, :] *
                 unique_probe[..., 0, m, :, :].conj())
    # TODO: Allow this kind of broadcasting inside the patch operator
    if probe_amp.shape[-3] == 1:
        probe_amp = cp.tile(probe_amp, (scan_.shape[-2], 1, 1))
    if psi_update_denominator is None:
        psi_update_denominator = cp.zeros(
            shape=psi.shape,
            dtype='complex64',
        )
    psi_update_denominator = op.diffraction.patch.adj(
        patches=probe_amp,
        images=psi_update_denominator,
        positions=scan_,
    )
    return psi_update_denominator


def _update_wavefront(data, varying_probe, scan, psi, op):

    farplane = op.fwd(probe=varying_probe, scan=scan, psi=psi)
    intensity = cp.sum(
        cp.square(cp.abs(farplane)),
        axis=list(range(1, farplane.ndim - 2)),
    )
    cost = op.propagation.cost(data, intensity)
    logger.info('%10s cost is %+12.5e', 'farplane', cost)
    farplane = -op.propagation.grad(data, farplane, intensity)

    farplane = op.propagation.adj(farplane, overwrite=True)

    pad, end = op.diffraction.pad, op.diffraction.end
    return farplane[..., pad:end, pad:end], cost


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
    num_batch,
    psi_update_denominator,
    *,
    object_options,
    probe_options,
    algorithm_options,
):

    patches = comm.pool.map(_get_patches, nearplane, psi, scan_, op=op)

    for m in range(probe[0].shape[-3]):

        (
            diff,
            grad_probe,
            common_grad_psi,
            common_grad_probe,
            _,
            probe_update_denominator,
        ) = (list(a) for a in zip(*comm.pool.map(
            _get_nearplane_gradients,
            nearplane,
            psi,
            scan_,
            unique_probe,
            patches,
            op=op,
            m=m,
            recover_psi=recover_psi,
            recover_probe=recover_probe,
        )))

        if recover_psi:
            if comm.use_mpi:
                common_grad_psi = comm.Allreduce(common_grad_psi)
            else:
                common_grad_psi = comm.pool.allreduce(common_grad_psi)

        if recover_probe:
            if comm.use_mpi:
                common_grad_probe = comm.Allreduce(common_grad_probe)
                probe_update_denominator = comm.Allreduce(
                    probe_update_denominator)
            else:
                common_grad_probe = comm.pool.allreduce(common_grad_probe)
                probe_update_denominator = comm.pool.allreduce(
                    probe_update_denominator)

        (
            common_grad_psi,
            common_grad_probe,
            dOP,
            dPO,
            A1,
            A4,
        ) = (list(a) for a in zip(*comm.pool.map(
            _precondition_nearplane_gradients,
            nearplane,
            scan_,
            unique_probe,
            probe,
            common_grad_psi,
            common_grad_probe,
            psi_update_denominator,
            probe_update_denominator,
            patches,
            op=op,
            m=m,
            recover_psi=recover_psi,
            recover_probe=recover_probe,
            probe_options=probe_options,
        )))

        if recover_psi:
            if comm.use_mpi:
                delta = comm.Allreduce_mean(A1, axis=-3)
            else:
                delta = comm.pool.reduce_mean(A1, axis=-3)
            A1 = comm.pool.map(_A_diagonal_dominant, A1,
                               comm.pool.bcast([delta]))

        if recover_probe:
            if comm.use_mpi:
                delta = comm.Allreduce_mean(A4, axis=-3)
            else:
                delta = comm.pool.reduce_mean(A4, axis=-3)
            A4 = comm.pool.map(_A_diagonal_dominant, A4,
                               comm.pool.bcast([delta]))

        if m == 0 and (recover_probe or recover_psi):
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
            if comm.use_mpi:
                weighted_step_psi[0] = comm.Allreduce_mean(
                    weighted_step_psi,
                    axis=-5,
                )[..., 0, 0, 0]
                weighted_step_probe[0] = comm.Allreduce_mean(
                    weighted_step_probe,
                    axis=-5,
                )
            else:
                weighted_step_psi[0] = comm.pool.reduce_mean(
                    weighted_step_psi,
                    axis=-5,
                )[..., 0, 0, 0]
                weighted_step_probe[0] = comm.pool.reduce_mean(
                    weighted_step_probe,
                    axis=-5,
                )

        if m == 0 and recover_probe and eigen_weights[0] is not None:
            logger.info('Updating eigen probes')

            eigen_weights = comm.pool.map(
                _get_coefs_intensity,
                eigen_weights,
                diff,
                probe,
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

            if eigen_probe[0] is not None and m < eigen_probe[0].shape[-3]:
                assert eigen_weights[0].shape[
                    -2] == eigen_probe[0].shape[-4] + 1
                for n in range(1, eigen_probe[0].shape[-4] + 1):

                    (
                        eigen_probe,
                        eigen_weights,
                    ) = tike.ptycho.probe.update_eigen_probe(
                        comm,
                        R,
                        eigen_probe,
                        eigen_weights,
                        patches,
                        diff,
                        β=min(0.1, 1.0 / num_batch),
                        c=n,
                        m=m,
                    )

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
            # (27b) Object update
            dpsi = (weighted_step_psi[0] /
                    probe[0].shape[-3]) * common_grad_psi[0]

            if algorithm_options.batch_method != 'cluster_compact':
                if object_options.use_adaptive_moment:
                    (
                        dpsi,
                        object_options.v,
                        object_options.m,
                    ) = tike.opt.momentum(
                        g=dpsi,
                        v=object_options.v,
                        m=object_options.m,
                        vdecay=object_options.vdecay,
                        mdecay=object_options.mdecay,
                    )
                psi[0] = psi[0] + dpsi
                psi = comm.pool.bcast([psi[0]])
            else:
                object_options.combined_update += dpsi

        if recover_probe:
            dprobe = weighted_step_probe[0] * common_grad_probe[0]
            if probe_options.use_adaptive_moment:
                if probe_options.m is None:
                    probe_options.m = cp.zeros_like(probe[0])
                (
                    dprobe,
                    probe_options.v,
                    probe_options.m[..., [m], :, :],
                ) = tike.opt.momentum(
                    g=dprobe,
                    v=probe_options.v,
                    m=probe_options.m[..., [m], :, :],
                    vdecay=probe_options.vdecay,
                    mdecay=probe_options.mdecay,
                )

            # (27a) Probe update
            probe[0][..., [m], :, :] += dprobe
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


def _get_patches(nearplane, psi, scan, *, op):
    patches = op.diffraction.patch.fwd(
        patches=cp.zeros(
            nearplane[..., 0, 0, :, :].shape,
            dtype='complex64',
        ),
        images=psi,
        positions=scan,
    )[..., None, None, :, :]
    return patches


def _get_nearplane_gradients(
    nearplane,
    psi,
    scan_,
    unique_probe,
    patches,
    *,
    op,
    m,
    recover_psi,
    recover_probe,
):

    diff = nearplane[..., [m], :, :]

    if __debug__:
        logger.debug('%10s cost is %+12.5e', 'nearplane',
                     tike.linalg.norm(diff))

    if recover_psi:
        # (24b)
        grad_psi = cp.conj(unique_probe[..., [m], :, :]) * diff
        # (25b) Common object gradient.
        common_grad_psi = op.diffraction.patch.adj(
            patches=grad_psi[..., 0, 0, :, :],
            images=cp.zeros(psi.shape, dtype='complex64'),
            positions=scan_,
        )
    else:
        common_grad_psi = None
    psi_update_denominator = None

    if recover_probe:
        # (24a)
        grad_probe = cp.conj(patches) * diff
        # (25a) Common probe gradient. Use simple average instead of
        # division as described in publication because that's what
        # ptychoshelves does
        common_grad_probe = cp.sum(
            grad_probe,
            axis=-5,
            keepdims=True,
        )
        # Sum the amplitude of all the object patches to precondition the probe
        # update.
        probe_update_denominator = cp.sum(
            patches * patches.conj(),
            axis=-5,
            keepdims=True,
        )
    else:
        grad_probe = None
        common_grad_probe = None
        probe_update_denominator = None

    return (
        diff,
        grad_probe,
        common_grad_psi,
        common_grad_probe,
        psi_update_denominator,
        probe_update_denominator,
    )


def _precondition_nearplane_gradients(
    nearplane,
    scan_,
    unique_probe,
    probe,
    common_grad_psi,
    common_grad_probe,
    psi_update_denominator,
    probe_update_denominator,
    patches,
    *,
    op,
    m,
    recover_psi,
    recover_probe,
    alpha=0.05,
    probe_options,
):

    diff = nearplane[..., [m], :, :]

    eps = 1e-9 / (diff.shape[-2] * diff.shape[-1])

    if recover_psi:
        common_grad_psi /= ((1 - alpha) * psi_update_denominator +
                            alpha * psi_update_denominator.max(
                                axis=(-2, -1),
                                keepdims=True,
                            ))
        dOP = op.diffraction.patch.fwd(
            patches=cp.zeros(patches.shape, dtype='complex64')[..., 0, 0, :, :],
            images=common_grad_psi,
            positions=scan_,
        )[..., None, None, :, :] * unique_probe[..., [m], :, :]
        A1 = cp.sum((dOP * dOP.conj()).real + eps, axis=(-2, -1))
    else:
        common_grad_psi = None
        dOP = None
        A1 = None

    if recover_probe:

        b = tike.ptycho.probe.finite_probe_support(
            unique_probe[..., [m], :, :],
            p=probe_options.probe_support,
            radius=probe_options.probe_support_radius,
            degree=probe_options.probe_support_degree,
        )

        common_grad_probe = (common_grad_probe - b * probe[..., [m], :, :]) / (
            (1 - alpha) * probe_update_denominator +
            alpha * probe_update_denominator.max(
                axis=(-2, -1),
                keepdims=True,
            ) + b)

        dPO = common_grad_probe * patches
        A4 = cp.sum((dPO * dPO.conj()).real + eps, axis=(-2, -1))
    else:
        common_grad_probe = None
        dPO = None
        A4 = None

    return (
        common_grad_psi,
        common_grad_probe,
        dOP,
        dPO,
        A1,
        A4,
    )


def _A_diagonal_dominant(A, delta):
    A += 0.5 * delta
    return A


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
        step = 0.9 * cp.maximum(0, x1[..., None, None].real)

        # (27b) Object update
        weighted_step_psi = cp.mean(step, keepdims=True, axis=-5)

    if recover_probe:
        step = 0.9 * cp.maximum(0, x2[..., None, None].real)

        weighted_step_probe = cp.mean(step, axis=-5, keepdims=True)
    else:
        weighted_step_probe = None

    return weighted_step_psi, weighted_step_probe


def _get_coefs_intensity(weights, xi, P, O, m):
    OP = O * P[..., [m], :, :]
    num = cp.sum(cp.real(cp.conj(OP) * xi), axis=(-1, -2))
    den = cp.sum(cp.abs(OP)**2, axis=(-1, -2))
    weights[..., 0:1, [m]] = weights[..., 0:1, [m]] + 0.1 * num / den
    return weights


def _get_residuals(grad_probe, grad_probe_mean):
    return grad_probe - grad_probe_mean


def _update_residuals(R, eigen_probe, axis, c, m):
    R -= tike.linalg.projection(
        R,
        eigen_probe[..., c:c + 1, m:m + 1, :, :],
        axis=axis,
    )
    return R


def _update_position(
    position_options,
    diff,
    patches,
    scan,
    unique_probe,
    m,
    op,
    *,
    alpha=0.05,
    max_shift=1,
):
    main_probe = unique_probe[..., m:m + 1, :, :]

    # According to the manuscript, we can either shift the probe or the object
    # and they are equivalent (in theory). Here we shift the object because
    # that is what ptychoshelves does.
    grad_y, grad_x = tike.ptycho.position._image_grad(op, patches)

    numerator = cp.sum(cp.real(diff * cp.conj(grad_x * main_probe)),
                       axis=(-2, -1))
    denominator = cp.sum(cp.abs(grad_x * main_probe)**2, axis=(-2, -1))
    step_x = numerator / (
        (1 - alpha) * denominator + alpha * max(denominator.max(), 1e-6))

    numerator = cp.sum(cp.real(diff * cp.conj(grad_y * main_probe)),
                       axis=(-2, -1))
    denominator = cp.sum(cp.abs(grad_y * main_probe)**2, axis=(-2, -1))
    step_y = numerator / (
        (1 - alpha) * denominator + alpha * max(denominator.max(), 1e-6))

    step_x = step_x[..., 0, 0]
    step_y = step_y[..., 0, 0]

    # Momentum
    if position_options.use_adaptive_moment:
        logger.info(
            "position correction with ADAptive Momemtum acceleration enabled.")
        step_x, position_options.vx, position_options.mx = tike.opt.adam(
            step_x,
            position_options.vx,
            position_options.mx,
            vdecay=position_options.vdecay,
            mdecay=position_options.mdecay)
        step_y, position_options.vy, position_options.my = tike.opt.adam(
            step_y,
            position_options.vy,
            position_options.my,
            vdecay=position_options.vdecay,
            mdecay=position_options.mdecay)

    # Step limit for stability
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
    logger.info('position update norm is %+.3e', tike.linalg.norm(step_x))

    # print(cp.abs(step_x) > 0.5)
    # print(cp.abs(step_y) > 0.5)

    scan[..., 0] -= step_y
    scan[..., 1] -= step_x

    return scan, position_options


def _mad(x, **kwargs):
    """Return the mean absolute deviation around the median."""
    return cp.mean(cp.abs(x - cp.median(x, **kwargs)), **kwargs)
