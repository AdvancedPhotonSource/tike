import logging
import typing

import cupy as cp
import numpy.typing as npt

import tike.communicators
import tike.linalg
import tike.operators
import tike.opt
import tike.random
import tike.ptycho.position
import tike.ptycho.probe
import tike.ptycho.object
import tike.ptycho.exitwave
import tike.precision

from .options import *

logger = logging.getLogger(__name__)


def lstsq_grad(
    op: tike.operators.Ptycho,
    comm: tike.communicators.Comm,
    data: typing.List[npt.NDArray],
    batches: typing.List[npt.NDArray[cp.intc]],
    *,
    parameters: PtychoParameters,
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
    exitwave_options = parameters.exitwave_options

    eigen_probe = parameters.eigen_probe
    eigen_weights = parameters.eigen_weights

    if eigen_probe is None:
        beigen_probe = [None] * comm.pool.num_workers
    else:
        beigen_probe = eigen_probe

    if object_options is not None:
        if algorithm_options.batch_method == 'compact':
            object_options.combined_update = cp.zeros_like(psi[0])

    if probe_options is not None:
        probe_options.probe_update_sum = cp.zeros_like(probe[0])

    if parameters.algorithm_options.batch_method == 'compact':
        order = range
    else:
        order = tike.random.randomizer_np.permutation

    batch_cost = []
    beta_object = []
    beta_probe = []
    for n in order(len(batches[0])):

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

        nearplane, _, costs = zip(*comm.pool.map(
            _update_wavefront,
            bdata,
            unique_probe,
            bscan,
            psi,
            exitwave_options.measured_pixels,
            exitwave_options=exitwave_options,
            op=op,
        ))

        for c in costs:
            batch_cost = batch_cost + c.tolist()

        (
            psi,
            probe,
            beigen_probe,
            beigen_weights,
            bscan,
            bposition_options,
            bbeta_object,
            bbeta_probe,
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

        beta_object.append(bbeta_object)
        beta_probe.append(bbeta_probe)

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

    algorithm_options.costs.append(batch_cost)

    if object_options and algorithm_options.batch_method == 'compact':
        object_update_precond = _precondition_object_update(
            object_options.combined_update,
            object_options.preconditioner[0],
        )

        # (27b) Object update
        beta_object = (cp.mean(cp.stack(beta_object)) / probe[0].shape[-3])
        dpsi = beta_object * object_update_precond
        psi[0] = psi[0] + dpsi

        if object_options.use_adaptive_moment:
            (
                dpsi,
                object_options.v,
                object_options.m,
            ) = _momentum_checked(
                g=dpsi,
                v=object_options.v,
                m=object_options.m,
                mdecay=object_options.mdecay,
                errors=list(np.mean(x) for x in algorithm_options.costs[-3:]),
                beta=beta_object,
                memory_length=3,
            )
            weight = object_options.preconditioner[0]
            weight = weight / (0.1 * weight.max() + weight)
            psi[0] = psi[0] + weight * dpsi

        psi = comm.pool.bcast([psi[0]])

    if probe_options:
        if probe_options.use_adaptive_moment:
            beta_probe = cp.mean(cp.stack(beta_probe))
            dprobe = probe_options.probe_update_sum
            if probe_options.v is None:
                probe_options.v = np.zeros_like(
                    dprobe,
                    shape=(3, *dprobe.shape),
                )
            if probe_options.m is None:
                probe_options.m = np.zeros_like(dprobe,)
            # ptychoshelves only applies momentum to the main probe
            mode = 0
            (
                d,
                probe_options.v[..., mode, :, :],
                probe_options.m[..., mode, :, :],
            ) = _momentum_checked(
                g=dprobe[..., mode, :, :],
                v=probe_options.v[..., mode, :, :],
                m=probe_options.m[..., mode, :, :],
                mdecay=probe_options.mdecay,
                errors=list(np.mean(x) for x in algorithm_options.costs[-3:]),
                beta=beta_probe,
                memory_length=3,
            )
            probe[0][..., mode, :, :] = probe[0][..., mode, :, :] + d
            probe = comm.pool.bcast([probe[0]])

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


def _update_wavefront(
    data: npt.NDArray,
    varying_probe: npt.NDArray[cp.csingle],
    scan: npt.NDArray[cp.single],
    psi: npt.NDArray[cp.csingle],
    measured_pixels: npt.NDArray[cp.bool_],
    *,
    exitwave_options: ExitWaveOptions,
    op: tike.operators.Ptycho,
) -> typing.Tuple[npt.NDArray[cp.csingle], float]:

    farplane = op.fwd(probe=varying_probe, scan=scan, psi=psi)

    intensity = cp.sum(
        cp.square(cp.abs(farplane)),
        axis=list(range(1, farplane.ndim - 2)),
    )

    costs = getattr(tike.operators, f'{exitwave_options.noise_model}_each_pattern')(
        data[:, measured_pixels][:, None, :],
        intensity[:, measured_pixels][:, None, :])

    cost = cp.mean(costs)

    farplane_opt = cp.empty_like(farplane)

    if exitwave_options.noise_model == 'poisson':

        xi = (1 - data / (intensity + 1e-9))[:, None, None, ...]
        grad_cost = farplane * xi

        step_length = cp.full(
            shape=(farplane.shape[0], 1, farplane.shape[2], 1, 1),
            fill_value=exitwave_options.step_length_start,
        )

        if exitwave_options.step_length_usemodes == 'dominant_mode':

            step_length = tike.ptycho.exitwave.poisson_steplength_dominant_mode(
                xi,
                intensity,
                data,
                measured_pixels,
                step_length,
                exitwave_options.step_length_weight,
            )

        else:

            step_length = tike.ptycho.exitwave.poisson_steplength_all_modes(
                xi,
                cp.square(cp.abs(farplane)),
                intensity,
                data,
                measured_pixels,
                step_length,
                exitwave_options.step_length_weight,
            )

        farplane_opt[..., measured_pixels] = (
            farplane -
            step_length * grad_cost)[..., measured_pixels]

    else:

        farplane_opt[..., measured_pixels] = (
            farplane * ((cp.sqrt(data) /
                         (cp.sqrt(intensity) + 1e-9))[..., None, None, :, :])
        )[..., measured_pixels]

    unmeasured_pixels = cp.logical_not(measured_pixels)
    farplane_opt[..., unmeasured_pixels] = farplane[
        ..., unmeasured_pixels] * exitwave_options.unmeasured_pixels_scaling

    farplane = farplane_opt - farplane

    farplane = op.propagation.adj(farplane, overwrite=True)

    pad, end = op.diffraction.pad, op.diffraction.end

    return farplane[..., pad:end, pad:end], cost, costs


def _update_nearplane(
    op: tike.operators.Ptycho,
    comm: tike.communicators.Comm,
    nearplane: typing.List[npt.NDArray[cp.csingle]],
    psi: typing.List[npt.NDArray[cp.csingle]],
    scan_: typing.List[npt.NDArray[cp.single]],
    probe: typing.List[npt.NDArray[cp.csingle]],
    unique_probe: typing.List[npt.NDArray[cp.csingle]],
    eigen_probe: typing.List[npt.NDArray[cp.csingle]],
    eigen_weights: typing.List[npt.NDArray[cp.single]],
    recover_psi: bool,
    recover_probe: bool,
    position_options: typing.Union[PositionOptions, None],
    num_batch: int,
    psi_update_denominator: npt.NDArray[cp.csingle],
    *,
    object_options: typing.Union[ObjectOptions, None],
    probe_options: typing.Union[ProbeOptions, None],
    algorithm_options: LstsqOptions,
):

    patches = comm.pool.map(_get_patches, nearplane, psi, scan_, op=op)

    for m in range(probe[0].shape[-3]):

        (
            diff,
            probe_update,
            object_upd_sum,
            m_probe_update,
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
            object_upd_sum = comm.Allreduce(object_upd_sum)

        if recover_probe:
            m_probe_update = comm.Allreduce_mean(
                m_probe_update,
                axis=-5,
            )
            probe_update_denominator = comm.Allreduce_mean(
                probe_update_denominator,
                axis=-5,
            )

        (
            object_update_precond,
            m_probe_update,
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
            object_upd_sum,
            m_probe_update,
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
            delta = comm.Allreduce_mean(A1, axis=-3)
            A1 = comm.pool.map(_A_diagonal_dominant, A1,
                               comm.pool.bcast([delta]))

        if recover_probe:
            delta = comm.Allreduce_mean(A4, axis=-3)
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
            weighted_step_psi[0] = comm.Allreduce_mean(
                weighted_step_psi,
                axis=-5,
            )[..., 0, 0, 0]
            weighted_step_probe[0] = comm.Allreduce_mean(
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
                R = comm.pool.map(_get_residuals, probe_update, m_probe_update)

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
        if object_options is not None:
            if algorithm_options.batch_method != 'compact':
                # (27b) Object update
                dpsi = (weighted_step_psi[0] /
                        probe[0].shape[-3]) * object_update_precond[0]

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
                object_options.combined_update += object_upd_sum[0]

        if probe_options is not None:
            dprobe = weighted_step_probe[0] * m_probe_update[0]
            probe_options.probe_update_sum[..., [m], :, :] += dprobe / num_batch
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

    return (
        psi,
        probe,
        eigen_probe,
        eigen_weights,
        scan_,
        position_options,
        weighted_step_psi[0],
        weighted_step_probe[0],
    )


def _get_patches(
    nearplane: npt.NDArray[cp.csingle],
    psi: npt.NDArray[cp.csingle],
    scan: npt.NDArray[cp.single],
    *,
    op: tike.operators.Ptycho,
) -> npt.NDArray[cp.csingle]:
    patches = op.diffraction.patch.fwd(
        patches=cp.zeros_like(nearplane[..., 0, 0, :, :]),
        images=psi,
        positions=scan,
    )[..., None, None, :, :]
    return patches


def _get_nearplane_gradients(
    nearplane: npt.NDArray[cp.csingle],
    psi: npt.NDArray[cp.csingle],
    scan_: npt.NDArray[cp.single],
    unique_probe: npt.NDArray[cp.csingle],
    patches: npt.NDArray[cp.csingle],
    *,
    op: tike.operators.Ptycho,
    m: int,
    recover_psi: bool,
    recover_probe: bool,
):

    chi = nearplane[..., [m], :, :]

    if __debug__:
        logger.debug('%10s cost is %+12.5e', 'nearplane', tike.linalg.norm(chi))

    # Get update directions for each scan positions
    if recover_psi:
        # (24b)
        object_update_proj = cp.conj(unique_probe[..., [m], :, :]) * chi
        # (25b) Common object gradient.
        object_upd_sum = op.diffraction.patch.adj(
            patches=object_update_proj[..., 0, 0, :, :],
            images=cp.zeros_like(psi),
            positions=scan_,
        )
    else:
        object_upd_sum = None

    if recover_probe:
        # (24a)
        probe_update = cp.conj(patches) * chi
        # (25a) Common probe gradient. Use simple average instead of
        # division as described in publication because that's what
        # ptychoshelves does
        m_probe_update = cp.mean(
            probe_update,
            axis=-5,
            keepdims=True,
        )
        # Sum the amplitude of all the object patches to precondition the probe
        # update.
        probe_update_denominator = cp.mean(
            patches * patches.conj(),
            axis=-5,
            keepdims=True,
        )
    else:
        probe_update = None
        m_probe_update = None
        probe_update_denominator = None

    return (
        chi,
        probe_update,
        object_upd_sum,
        m_probe_update,
        None,
        probe_update_denominator,
    )


def _precondition_object_update(
    object_upd_sum: npt.NDArray[cp.csingle],
    psi_update_denominator: npt.NDArray[cp.csingle],
    alpha: float = 0.05,
) -> npt.NDArray[cp.csingle]:
    return object_upd_sum / cp.sqrt(
        cp.square((1 - alpha) * psi_update_denominator) +
        cp.square(alpha * psi_update_denominator.max(
            axis=(-2, -1),
            keepdims=True,
        )))


def _precondition_nearplane_gradients(
    nearplane,
    scan_,
    unique_probe,
    probe,
    object_upd_sum,
    m_probe_update,
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

    eps = op.xp.float32(1e-9) / (diff.shape[-2] * diff.shape[-1])

    if recover_psi:
        object_update_precond = _precondition_object_update(
            object_upd_sum,
            psi_update_denominator,
        )

        object_update_proj = op.diffraction.patch.fwd(
            patches=cp.zeros_like(patches[..., 0, 0, :, :]),
            images=object_update_precond,
            positions=scan_,
        )
        dOP = object_update_proj[..., None,
                                 None, :, :] * unique_probe[..., [m], :, :]

        A1 = cp.sum((dOP * dOP.conj()).real + eps, axis=(-2, -1))
    else:
        object_update_proj = None
        dOP = None
        A1 = None

    if recover_probe:

        # b0 = tike.ptycho.probe.finite_probe_support(
        #     unique_probe[..., [m], :, :],
        #     p=probe_options.probe_support,
        #     radius=probe_options.probe_support_radius,
        #     degree=probe_options.probe_support_degree,
        # )

        # b1 = probe_options.additional_probe_penalty * cp.linspace(
        #     0,
        #     1,
        #     probe[0].shape[-3],
        #     dtype=tike.precision.floating,
        # )[..., [m], None, None]

        # m_probe_update = (m_probe_update -
        #                   (b0 + b1) * probe[..., [m], :, :]) / (
        #                       (1 - alpha) * probe_update_denominator +
        #                       alpha * probe_update_denominator.max(
        #                           axis=(-2, -1),
        #                           keepdims=True,
        #                       ) + b0 + b1)

        dPO = m_probe_update * patches
        A4 = cp.sum((dPO * dPO.conj()).real + eps, axis=(-2, -1))
    else:
        dPO = None
        A4 = None

    return (
        object_update_precond,
        m_probe_update,
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
    else:
        x1 = None
        x2 = None

    if recover_psi:
        step = 0.9 * cp.maximum(0, x1[..., None, None].real)

        # (27b) Object update
        beta_object = cp.mean(step, keepdims=True, axis=-5)
    else:
        beta_object = None

    if recover_probe:
        step = 0.9 * cp.maximum(0, x2[..., None, None].real)

        beta_probe = cp.mean(step, axis=-5, keepdims=True)
    else:
        beta_probe = None

    return beta_object, beta_probe


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
    grad_x, grad_y = tike.ptycho.position.gaussian_gradient(patches)

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

    step = cp.stack((step_x, step_y), axis=-1)

    # Momentum
    if position_options.use_adaptive_moment:
        logger.info(
            "position correction with ADAptive Momemtum acceleration enabled.")
        step, position_options.v, position_options.m = tike.opt.adam(
            step,
            position_options.v,
            position_options.m,
            vdecay=position_options.vdecay,
            mdecay=position_options.mdecay,
        )

    scan -= step

    return scan, position_options


def _momentum_checked(
    g: npt.NDArray,
    v: typing.Union[None, npt.NDArray],
    m: typing.Union[None, npt.NDArray],
    mdecay: float,
    errors: typing.List[float],
    beta: float = 1.0,
    memory_length: int = 3,
    vdecay=None,
) -> typing.Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Momentum updates, but only if the cost function is trending downward.

    Parameters
    ----------
    previous_g (EPOCH, WIDTH, HEIGHT)
        The previous psi updates
    g (WIDTH, HEIGHT)
        The current psi update
    """
    m = np.zeros_like(g,) if m is None else m
    previous_g = np.zeros_like(
        g,
        shape=(memory_length, *g.shape),
    ) if v is None else v

    # Keep a running list of the update directions
    previous_g = np.roll(previous_g, shift=-1, axis=0)
    previous_g[-1] = g / tike.linalg.norm(g) * beta

    # Only apply momentum updates if the objective function is decreasing
    if (len(errors) > 2
            and max(errors[-3], errors[-2]) > min(errors[-2], errors[-1])):
        # Check that previous updates are moving in a similar direction
        previous_update_correlation = tike.linalg.inner(
            previous_g[:-1],
            previous_g[-1],
            axis=(-2, -1),
        ).real.flatten()
        if np.all(previous_update_correlation > 0):
            friction, _ = tike.opt.fit_line_least_squares(
                x=np.arange(len(previous_update_correlation) + 1),
                y=[
                    0,
                ] + np.log(previous_update_correlation).tolist(),
            )
            friction = 0.5 * max(-friction, 0)
            m = (1 - friction) * m + g
            return mdecay * m, previous_g, m

    return np.zeros_like(g), previous_g, m / 2
