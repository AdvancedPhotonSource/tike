import logging
import typing

import cupy as cp
import cupyx.scipy.stats
import numpy as np
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
    parameters: PtychoParameters,
    data: npt.NDArray,
    batches: typing.List[npt.NDArray[cp.intc]],
    streams: typing.List[cp.cuda.Stream],
    worker_index: int,
    *,
    op: tike.operators.Ptycho,
    epoch: int,
) -> PtychoParameters:
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
    scan = parameters.scan
    psi = parameters.psi
    probe = parameters.probe
    algorithm_options = parameters.algorithm_options
    eigen_weights = parameters.eigen_weights
    eigen_probe = parameters.eigen_probe
    measured_pixels = parameters.exitwave_options.measured_pixels
    exitwave_options = parameters.exitwave_options
    position_options = parameters.position_options
    object_options = parameters.object_options
    probe_options = parameters.probe_options
    recover_probe = probe_options is not None and epoch >= probe_options.update_start

    # CONVERSTION AREA ABOVE ---------------------------------------

    if parameters.algorithm_options.batch_method == 'compact':
        order = range
    else:
        order = tike.random.randomizer_np.permutation

    object_combined_update = cp.zeros_like(psi)
    probe_combined_update = cp.zeros_like(probe)
    position_update_numerator = None
    position_update_denominator = None

    batch_cost = cp.empty(algorithm_options.num_batch, dtype=tike.precision.floating)
    beta_object: typing.List[float] = []
    beta_probe: typing.List[float] = []
    for batch_index in order(algorithm_options.num_batch):
        (
            diff,
            unique_probe,
            probe_update,
            object_upd_sum,
            m_probe_update,
            costs,
            patches,
            position_update_numerator,
            position_update_denominator,
            position_options,
        ) = _get_nearplane_gradients(
            data,
            psi,
            scan,
            probe,
            eigen_probe,
            eigen_weights,
            batches,
            position_update_numerator,
            position_update_denominator,
            position_options,
            streams,
            measured_pixels,
            object_options.preconditioner,
            batch_index=batch_index,
            num_batch=algorithm_options.num_batch,
            exitwave_options=exitwave_options,
            op=op,
            recover_psi=object_options is not None,
            recover_probe=recover_probe,
            recover_positions=position_options is not None,
        )

        if recover_probe:
            m_probe_update = cp.mean(
                m_probe_update,
                axis=-5,
            )

            (
                eigen_probe,
                eigen_weights,
            ) = _update_nearplane(
                diff,
                probe_update,
                m_probe_update,
                probe,
                eigen_probe,
                eigen_weights,
                patches,
                batches,
                batch_index=batch_index,
                num_batch=algorithm_options.num_batch,
            )

        (
            object_update_precond,
            A1,
            A2,
            A4,
            b1,
            b2,
        ) = _precondition_nearplane_gradients(
            diff,
            scan,
            unique_probe,
            probe,
            object_upd_sum,
            m_probe_update,
            object_options.preconditioner,
            patches,
            batches,
            batch_index=batch_index,
            op=op,
            m=0,
            recover_psi=object_options is not None,
            recover_probe=recover_probe,
            probe_options=probe_options,
        )

        if object_options is not None:
            A1_delta = cp.mean(A1, axis=-3)
        else:
            A1_delta = None

        if recover_probe:
            A4_delta = cp.mean(A4, axis=-3)
        else:
            A4_delta = None

        (
            weighted_step_psi,
            weighted_step_probe,
        ) = _get_nearplane_steps(
            A1,
            A2,
            A4,
            b1,
            b2,
            A1_delta,
            A4_delta,
            recover_psi=object_options is not None,
            recover_probe=recover_probe,
            m=0,
        )

        if object_options is not None:
            bbeta_object = cp.mean(
                weighted_step_psi,
                axis=-5,
            )[..., 0, 0, 0]

        if recover_probe:
            bbeta_probe = cp.mean(
                weighted_step_probe,
                axis=-5,
            )

        # Update each direction
        if object_options is not None:
            if algorithm_options.batch_method != 'compact':
                # (27b) Object update
                dpsi = bbeta_object * object_update_precond

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
                psi = psi + dpsi
            else:
                object_combined_update += object_upd_sum

            beta_object.append(bbeta_object)

        if recover_probe:
            dprobe = bbeta_probe * m_probe_update
            probe_combined_update += dprobe / algorithm_options.num_batch
            # (27a) Probe update
            probe += dprobe

            beta_probe.append(bbeta_probe)

        batch_cost[batch_index] = cp.mean(costs)

    if (
        position_options is not None
        and position_update_numerator is not None
        and position_update_denominator is not None
    ):
        scan, position_options = _update_position(
            scan,
            position_options,
            position_update_numerator,
            position_update_denominator,
            epoch=epoch,
        )

    algorithm_options.costs.append([float(batch_cost.mean().get())])

    if object_options and algorithm_options.batch_method == 'compact':
        object_update_precond = _precondition_object_update(
            object_combined_update,
            object_options.preconditioner,
        )

        # (27b) Object update
        beta_object = cp.mean(cp.stack(beta_object))
        dpsi = beta_object * object_update_precond
        psi = psi + dpsi

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
                errors=list(float(x[worker_index]) for x in algorithm_options.costs[-3:]),
                beta=beta_object,
                memory_length=3,
            )
            weight = object_options.preconditioner
            weight = weight / (0.1 * weight.max() + weight)
            psi = psi + weight * dpsi

    if recover_probe:
        if probe_options.use_adaptive_moment:
            beta_probe = cp.mean(cp.stack(beta_probe))
            dprobe = probe_combined_update
            if probe_options.v is None:
                probe_options.v = cp.zeros_like(
                    dprobe,
                    shape=(3, *dprobe.shape),
                )
            if probe_options.m is None:
                probe_options.m = cp.zeros_like(dprobe,)
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
                errors=list(float(x[worker_index]) for x in algorithm_options.costs[-3:]),
                beta=beta_probe,
                memory_length=3,
            )
            probe[..., mode, :, :] = probe[..., mode, :, :] + d

    # CONVERSION AREA BELOW ----------------------

    parameters.scan = scan
    parameters.psi = psi
    parameters.probe = probe
    parameters.algorithm_options = algorithm_options
    parameters.eigen_weights = eigen_weights
    parameters.eigen_probe = eigen_probe
    parameters.exitwave_options = exitwave_options
    parameters.position_options = position_options
    parameters.object_options = object_options
    parameters.probe_options = probe_options

    return parameters


def _update_nearplane(
    diff: npt.NDArray[cp.csingle],
    probe_update: npt.NDArray[cp.csingle],
    m_probe_update: npt.NDArray[cp.csingle],
    probe: npt.NDArray[cp.csingle],
    eigen_probe: npt.NDArray[cp.csingle],
    eigen_weights: npt.NDArray[cp.single],
    patches: npt.NDArray[cp.csingle],
    batches,
    *,
    batch_index: int,
    num_batch: int,
):
    m = 0
    if eigen_weights is not None:
        eigen_weights = _get_coefs_intensity(
            eigen_weights,
            diff,
            probe,
            patches,
            batches,
            batch_index=batch_index,
            m=m,
        )

        # (30) residual probe updates
        if eigen_weights.shape[-2] > 1:
            R = _get_residuals(
                probe_update,
                m_probe_update,
                m=m,
            )

        if eigen_probe is not None and m < eigen_probe.shape[-3]:
            assert eigen_weights.shape[-2] == eigen_probe.shape[-4] + 1
            for eigen_index in range(1, eigen_probe.shape[-4] + 1):
                (
                    eigen_probe,
                    eigen_weights,
                ) = tike.ptycho.probe.update_eigen_probe(
                    R,
                    eigen_probe,
                    eigen_weights,
                    patches,
                    diff,
                    batches,
                    batch_index=batch_index,
                    β=min(0.1, 1.0 / num_batch),
                    c=eigen_index,
                    m=m,
                )

                if eigen_index + 1 < eigen_weights.shape[-2]:
                    # Subtract projection of R onto new probe from R
                    R = _update_residuals(
                        R,
                        eigen_probe,
                        batches,
                        batch_index=batch_index,
                        axis=(-2, -1),
                        c=eigen_index - 1,
                        m=m,
                    )

    return (
        eigen_probe,
        eigen_weights,
    )


def _get_nearplane_gradients(
    data: npt.NDArray,
    psi: npt.NDArray[cp.csingle],
    scan: npt.NDArray[cp.single],
    probe: npt.NDArray[cp.csingle],
    eigen_probe,
    eigen_weights,
    batches,
    position_update_numerator,
    position_update_denominator,
    position_options: PositionOptions,
    streams: typing.List[cp.cuda.Stream],
    measured_pixels: npt.NDArray,
    object_preconditioner: npt.NDArray[cp.csingle],
    *,
    batch_index: int,
    num_batch: int,
    op: tike.operators.Ptycho,
    recover_psi: bool,
    recover_probe: bool,
    recover_positions: bool,
    exitwave_options: ExitWaveOptions,
):
    batch_start = batches[batch_index][0]
    batch_size = len(batches[batch_index])

    # These variables are only as large as the batch
    bcosts = cp.empty(shape=batch_size, dtype=tike.precision.floating)
    bchi = cp.empty_like(
        probe,
        shape=(batch_size, 1, *probe.shape[-3:]),
    )
    bpatches = cp.empty_like(
        probe,
        shape=(batch_size, 1, 1, *probe.shape[-2:]),
    )
    bprobe_update = cp.empty_like(
        probe,
        shape=bchi.shape,
    )
    bunique_probe = cp.empty_like(
        probe,
        shape=(batch_size, 1, *probe.shape[-3:]),
    )

    # These variables are as large as the entire dataset
    m_probe_update = cp.zeros_like(probe)
    object_upd_sum = cp.zeros_like(psi)
    position_update_numerator = cp.empty_like(
        scan
    ) if position_update_numerator is None else position_update_numerator
    position_update_denominator = cp.empty_like(
        scan
    ) if position_update_denominator is None else position_update_denominator

    def keep_some_args_constant(
        ind_args,
        lo: int,
        hi: int,
    ) -> None:
        (data,) = ind_args
        nonlocal bchi, bunique_probe, bprobe_update, object_upd_sum
        nonlocal m_probe_update, bcosts, bpatches, position_update_numerator
        nonlocal position_update_denominator

        blo = lo - batch_start
        bhi = hi - batch_start

        bunique_probe[blo:bhi] = tike.ptycho.probe.get_varying_probe(
            probe,
            eigen_probe,
            eigen_weights[lo:hi] if eigen_weights is not None else None,
        )

        farplane = op.fwd(probe=bunique_probe[blo:bhi],
                          scan=scan[lo:hi],
                          psi=psi)
        intensity = cp.sum(
            cp.square(cp.abs(farplane)),
            axis=list(range(1, farplane.ndim - 2)),
        )
        bcosts[blo:bhi] = getattr(
            tike.operators, f'{exitwave_options.noise_model}_each_pattern')(
                data[:, measured_pixels][:, None, :],
                intensity[:, measured_pixels][:, None, :],
            )

        if exitwave_options.noise_model == 'poisson':

            xi = (1 - data / (intensity + 1e-9))[:, None, None, ...]
            grad_cost = farplane * xi

            step_length = cp.full(
                shape=(farplane.shape[0], 1, farplane.shape[2], 1, 1),
                fill_value=tike.precision.floating(exitwave_options.step_length_start),
                dtype=tike.precision.floating,
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

            farplane[..., measured_pixels] = (-step_length *
                                              grad_cost)[..., measured_pixels]

        else:

            farplane[..., measured_pixels] = -getattr(
                tike.operators, f'{exitwave_options.noise_model}_grad')(
                    data,
                    farplane,
                    intensity,
                )[..., measured_pixels]

        unmeasured_pixels = cp.logical_not(measured_pixels)
        farplane[..., unmeasured_pixels] *= (
            exitwave_options.unmeasured_pixels_scaling - 1.0)

        farplane = op.propagation.adj(farplane, overwrite=True)

        pad, end = op.diffraction.pad, op.diffraction.end
        bchi[blo:bhi] = farplane[..., pad:end, pad:end]

        # Get update directions for each scan positions
        if recover_psi:
            # (24b)
            object_update_proj = cp.conj(bunique_probe[blo:bhi]) * bchi[blo:bhi]
            # (25b) Common object gradient.
            object_upd_sum[0] = op.diffraction.patch.adj(
                patches=object_update_proj.reshape(
                    len(scan[lo:hi]) * bchi.shape[-3], *bchi.shape[-2:]),
                images=object_upd_sum[0],
                positions=scan[lo:hi],
                nrepeat=bchi.shape[-3],
            )
        else:
            object_upd_sum = None

        if recover_probe:
            bpatches[blo:bhi] = op.diffraction.patch.fwd(
                patches=cp.zeros_like(bchi[blo:bhi, ..., 0, 0, :, :]),
                images=psi[0],
                positions=scan[lo:hi],
            )[..., None, None, :, :]
            # (24a)
            bprobe_update[blo:bhi] = cp.conj(bpatches[blo:bhi]) * bchi[blo:bhi]
            # (25a) Common probe gradient. Use simple average instead of
            # division as described in publication because that's what
            # ptychoshelves does
            m_probe_update += cp.sum(
                bprobe_update[blo:bhi],
                axis=-5,
                keepdims=True,
            )
        else:
            bprobe_update = None
            m_probe_update = None
            bpatches = None

        if position_options:
            m = 0
            grad_x, grad_y = tike.ptycho.position.gaussian_gradient(
                bpatches[blo:bhi],
                sigma=0.333,
            )
            crop = probe.shape[-1] // 4
            position_update_numerator[lo:hi, ..., 0] = cp.sum(
                cp.real(
                    cp.conj(grad_x[..., crop:-crop, crop:-crop] *
                            bunique_probe[blo:bhi, ..., m:m + 1, crop:-crop,
                                          crop:-crop]) *
                    bchi[blo:bhi, ..., m:m + 1, crop:-crop, crop:-crop]),
                axis=(-4, -3, -2, -1),
            )
            position_update_denominator[lo:hi, ..., 0] = cp.sum(
                cp.abs(grad_x[..., crop:-crop, crop:-crop] *
                       bunique_probe[blo:bhi, ..., m:m + 1, crop:-crop,
                                     crop:-crop])**2,
                axis=(-4, -3, -2, -1),
            )
            position_update_numerator[lo:hi, ..., 1] = cp.sum(
                cp.real(
                    cp.conj(grad_y[..., crop:-crop, crop:-crop] *
                            bunique_probe[blo:bhi, ..., m:m + 1, crop:-crop,
                                          crop:-crop]) *
                    bchi[blo:bhi, ..., m:m + 1, crop:-crop, crop:-crop]),
                axis=(-4, -3, -2, -1),
            )
            position_update_denominator[lo:hi, ..., 1] = cp.sum(
                cp.abs(grad_y[..., crop:-crop, crop:-crop] *
                       bunique_probe[blo:bhi, ..., m:m + 1, crop:-crop,
                                     crop:-crop])**2,
                axis=(-4, -3, -2, -1),
            )

    tike.communicators.stream.stream_and_modify2(
        f=keep_some_args_constant,
        ind_args=[
            data,
        ],
        streams=streams,
        lo=batch_start,
        hi=batch_start + batch_size,
    )

    return (
        bchi,
        bunique_probe,
        bprobe_update,
        object_upd_sum,
        m_probe_update / num_batch if m_probe_update is not None else None,
        bcosts,
        bpatches,
        position_update_numerator,
        position_update_denominator,
        position_options,
    )


def _precondition_object_update(
    object_upd_sum: npt.NDArray[cp.csingle],
    psi_update_denominator: npt.NDArray[cp.csingle],
    alpha: float = 0.05,
) -> npt.NDArray[cp.csingle]:
    return object_upd_sum / cp.sqrt(
        cp.square((1 - alpha) * psi_update_denominator) +
        cp.square(alpha * cp.amax(
            psi_update_denominator,
            axis=(-2, -1),
            keepdims=True,
        )))


def _precondition_nearplane_gradients(
    nearplane,
    scan,
    unique_probe,
    probe,
    object_upd_sum,
    m_probe_update,
    psi_update_denominator,
    patches,
    batches,
    *,
    batch_index: int,
    op,
    m,
    recover_psi,
    recover_probe,
    alpha=0.05,
    probe_options,
):
    lo = batches[batch_index][0]
    hi = lo + len(batches[batch_index])

    eps = op.xp.float32(1e-9) / (nearplane.shape[-2] * nearplane.shape[-1])

    A1 = None
    A2 = None
    A4 = None
    b1 = None
    b2 = None
    dOP = None
    dPO = None
    object_update_proj = None

    if recover_psi:
        object_update_precond = _precondition_object_update(
            object_upd_sum,
            psi_update_denominator,
        )

        object_update_proj = op.diffraction.patch.fwd(
            patches=cp.zeros_like(nearplane[..., 0, 0, :, :]),
            images=object_update_precond[0],
            positions=scan[lo:hi],
        )
        dOP = object_update_proj[..., None,
                                 None, :, :] * unique_probe[..., m:m + 1, :, :]

        A1 = cp.sum((dOP * dOP.conj()).real + eps, axis=(-2, -1))

    if recover_probe:
        dPO = m_probe_update[..., m:m + 1, :, :] * patches
        A4 = cp.sum((dPO * dPO.conj()).real + eps, axis=(-2, -1))

    if recover_psi and recover_probe:
        b1 = cp.sum((dOP.conj() * nearplane[..., m:m + 1, :, :]).real,
                    axis=(-2, -1))
        b2 = cp.sum((dPO.conj() * nearplane[..., m:m + 1, :, :]).real,
                    axis=(-2, -1))
        A2 = cp.sum((dOP * dPO.conj()), axis=(-2, -1))
    elif recover_psi:
        b1 = cp.sum((dOP.conj() * nearplane[..., m:m + 1, :, :]).real,
                    axis=(-2, -1))
    elif recover_probe:
        b2 = cp.sum((dPO.conj() * nearplane[..., m:m + 1, :, :]).real,
                    axis=(-2, -1))

    return (
        object_update_precond,
        A1,
        A2,
        A4,
        b1,
        b2,
    )


def _get_nearplane_steps(A1, A2, A4, b1, b2, A1_delta, A4_delta, recover_psi,
                         recover_probe, m):

    if recover_psi:
        A1 += 0.5 * A1_delta
    if recover_probe:
        A4 += 0.5 * A4_delta

    # (22) Use least-squares to find the optimal step sizes simultaneously
    if recover_psi and recover_probe:
        A3 = A2.conj()
        determinant = A1 * A4 - A2 * A3
        x1 = -cp.conj(A2 * b2 - A4 * b1) / determinant
        x2 = cp.conj(A1 * b2 - A3 * b1) / determinant
    elif recover_psi:
        x1 = b1 / A1
    elif recover_probe:
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


def _get_coefs_intensity(weights, xi, P, O, batches, *, batch_index, m):
    """
    Parameters
    ----------
    weights : (B, C, M)
    xi :      (B, 1, M, H, W)
    P :       (B, 1, M, H, W)
    O :       (B, 1, 1, H, W)
    """
    lo = batches[batch_index][0]
    hi = lo + len(batches[batch_index])
    OP = O * P[:, :, m:m + 1, :, :]
    num = cp.sum(cp.real(cp.conj(OP) * xi[:, :, m:m + 1, :, :]), axis=(-1, -2))
    den = cp.sum(cp.abs(OP)**2, axis=(-1, -2))
    weights[lo:hi, 0:1, m:m + 1] += 0.1 * num / den
    return weights


def _get_residuals(grad_probe, grad_probe_mean, m):
    """
    Parameters
    ----------
    grad_probe :      (B, 1, M, H, W)
    grad_probe_mean : (1, 1, M, H, W)
    """
    return grad_probe[..., m:m + 1, :, :] - grad_probe_mean[..., m:m + 1, :, :]


def _update_residuals(R, eigen_probe, batches, *, batch_index, axis, c, m):
    """
    Parameters
    ----------
    R :           (B, 1, 1, H, W)
    eigen_probe : (1, C, M, H, W)
    """
    R -= tike.linalg.projection(
        R,
        eigen_probe[:, c:c + 1, m:m + 1, :, :],
        axis=axis,
    )
    return R


def _update_position(
    scan: npt.NDArray,
    position_options: PositionOptions,
    position_update_numerator: npt.NDArray,
    position_update_denominator: npt.NDArray,
    *,
    alpha=0.05,
    max_shift=1,
    epoch=0,
):
    if epoch < position_options.update_start:
        return scan, position_options

    step = (position_update_numerator) / (
        (1 - alpha) * position_update_denominator +
        alpha * max(position_update_denominator.max(), 1e-6))

    if position_options.update_magnitude_limit > 0:
        step = cp.clip(
            step,
            a_min=-position_options.update_magnitude_limit,
            a_max=position_options.update_magnitude_limit,
        )

    # Remove outliars and subtract the mean
    step = step - cupyx.scipy.stats.trim_mean(step, 0.05)

    if position_options.use_adaptive_moment:
        (
            step,
            position_options.v,
            position_options.m,
        ) = tike.opt.adam(
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
