import logging
import typing

import cupy as cp
import cupyx.scipy.stats
import numpy.typing as npt
import numpy as np

import tike.communicators
import tike.linalg
import tike.operators
import tike.opt
import tike.ptycho.object
import tike.ptycho.position
import tike.ptycho.probe
import tike.ptycho.exitwave
import tike.precision
import tike.random

from .options import (
    ExitWaveOptions,
    ObjectOptions,
    PositionOptions,
    ProbeOptions,
    PtychoParameters,
    RpieOptions,
)
from .lstsq import _momentum_checked

logger = logging.getLogger(__name__)


def rpie(
    parameters: PtychoParameters,
    data: npt.NDArray,
    batches: typing.List[npt.NDArray[cp.intc]],
    streams: typing.List[cp.cuda.Stream],
    *,
    op: tike.operators.Ptycho,
    epoch: int,
) -> PtychoParameters:
    """Solve the ptychography problem using regularized ptychographical engine.

    The rPIE update direction can be shown to be equivalent to a conventional
    gradient descent direction but rescaled by the preconditioner term. i.e. If
    the rPIE step size (alpha) is 0 and the preconditioner is zero, we have the
    vanilla gradient descent direction.

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
    result : :py:class:`tike.ptycho.solvers.PtychoParameters`
        An object which contains the updated reconstruction parameters.

    References
    ----------
    Maiden, Andrew M., and John M. Rodenburg. 2009. “An Improved
    Ptychographical Phase Retrieval Algorithm for Diffractive Imaging.”
    Ultramicroscopy 109 (10): 1256–62.
    https://doi.org/10.1016/j.ultramic.2009.05.012.

    Andrew Maiden, Daniel Johnson, and Peng Li, "Further improvements to the
    ptychographical iterative engine," Optica 4, 736-745 (2017)
    https://doi.org/10.1364/OPTICA.4.000736

    .. seealso:: :py:mod:`tike.ptycho`

    """
    if parameters.algorithm_options.batch_method == 'compact':
        order = range
    else:
        order = tike.random.randomizer_np.permutation

    psi_update_numerator: None | cp.ndarray = None
    probe_update_numerator: None | cp.ndarray = None
    position_update_numerator: None | cp.ndarray = None
    position_update_denominator: None | cp.ndarray = None

    for n in order(parameters.algorithm_options.num_batch):
        (
            cost,
            psi_update_numerator,
            probe_update_numerator,
            position_update_numerator,
            position_update_denominator,
            parameters,
        ) = _get_nearplane_gradients(
            data,
            parameters,
            psi_update_numerator,
            probe_update_numerator,
            position_update_numerator,
            position_update_denominator,
            batches,
            streams,
            n=n,
            op=op,
            epoch=epoch,
        )

        if parameters.algorithm_options.batch_method != "compact":
            parameters = _update(
                parameters,
                psi_update_numerator,
                probe_update_numerator,
                recover_probe=parameters.probe_options.update_start >= epoch,
            )
            psi_update_numerator = None
            probe_update_numerator = None

    parameters.algorithm_options.costs.append([cost])

    if parameters.position_options is not None:
        (
            parameters.scan,
            parameters.position_options,
        ) = _update_position(
            parameters.scan,
            parameters.position_options,
            position_update_numerator,
            position_update_denominator,
            max_shift=parameters.probe.shape[-1] * 0.1,
            alpha=parameters.algorithm_options.alpha,
            epoch=epoch,
        )

    if parameters.algorithm_options.batch_method == "compact":
        parameters = _update(
            parameters,
            psi_update_numerator,
            probe_update_numerator,
            recover_probe=parameters.probe_options.update_start >= epoch,
            errors=list(
                float(np.mean(x)) for x in parameters.algorithm_options.costs[-3:]
            ),
        )

    if parameters.eigen_weights is not None:
        parameters.eigen_weights = _normalize_eigen_weights(
            parameters.eigen_weights,
        )

    return parameters


def _normalize_eigen_weights(eigen_weights):
    return eigen_weights / tike.linalg.mnorm(
        eigen_weights,
        axis=(-3),
        keepdims=True,
    )


def _update(
    parameters: PtychoParameters,
    psi_update_numerator: npt.NDArray[cp.csingle],
    probe_update_numerator: npt.NDArray[cp.csingle],
    recover_probe: bool,
    errors: typing.Union[None, typing.List[float]] = None,
) -> PtychoParameters:
    if parameters.object_options:
        dpsi = psi_update_numerator / (
            (1 - parameters.algorithm_options.alpha)
            * parameters.object_options.preconditioner
            + parameters.algorithm_options.alpha
            * parameters.object_options.preconditioner.max(
                axis=(-2, -1),
                keepdims=True,
            )
        )
        if parameters.object_options.use_adaptive_moment:
            if errors is not None:
                (
                    dpsi,
                    parameters.object_options.v,
                    parameters.object_options.m,
                ) = _momentum_checked(
                    g=dpsi,
                    v=parameters.object_options.v,
                    m=parameters.object_options.m,
                    mdecay=parameters.object_options.mdecay,
                    errors=errors,
                    memory_length=3,
                )
            else:
                (
                    dpsi,
                    parameters.object_options.v,
                    parameters.object_options.m,
                ) = tike.opt.adam(
                    g=dpsi,
                    v=parameters.object_options.v,
                    m=parameters.object_options.m,
                    vdecay=parameters.object_options.vdecay,
                    mdecay=parameters.object_options.mdecay,
                )
        parameters.psi = parameters.psi + dpsi

    if recover_probe and parameters.probe_options is not None:
        b0 = tike.ptycho.probe.finite_probe_support(
            parameters.probe,
            p=parameters.probe_options.probe_support,
            radius=parameters.probe_options.probe_support_radius,
            degree=parameters.probe_options.probe_support_degree,
        )
        b1 = (
            parameters.probe_options.additional_probe_penalty
            * cp.linspace(
                start=0,
                stop=1,
                num=parameters.probe.shape[-3],
                dtype="float32",
            )[..., None, None]
        )
        dprobe = (probe_update_numerator - (b1 + b0) * parameters.probe) / (
            (1 - parameters.algorithm_options.alpha)
            * parameters.probe_options.preconditioner
            + parameters.algorithm_options.alpha
            * parameters.probe_options.preconditioner.max(
                axis=(-2, -1),
                keepdims=True,
            )
            + b0
            + b1
        )
        if parameters.probe_options.use_adaptive_moment:
            # ptychoshelves only applies momentum to the main probe
            mode = 0
            if errors:
                (
                    dprobe[0, 0, mode, :, :],
                    parameters.probe_options.v,
                    parameters.probe_options.m,
                ) = _momentum_checked(
                    g=dprobe[0, 0, mode, :, :],
                    v=parameters.probe_options.v,
                    m=parameters.probe_options.m,
                    mdecay=parameters.probe_options.mdecay,
                    errors=errors,
                    memory_length=3,
                )
            else:
                (
                    dprobe[0, 0, mode, :, :],
                    parameters.probe_options.v,
                    parameters.probe_options.m,
                ) = tike.opt.adam(
                    g=dprobe[0, 0, mode, :, :],
                    v=parameters.probe_options.v,
                    m=parameters.probe_options.m,
                    vdecay=parameters.probe_options.vdecay,
                    mdecay=parameters.probe_options.mdecay,
                )
        parameters.probe = parameters.probe + dprobe

    return parameters


def _get_nearplane_gradients(
    data: npt.NDArray,
    parameters: PtychoParameters,
    psi_update_numerator: typing.Union[None, npt.NDArray],
    probe_update_numerator: typing.Union[None, npt.NDArray],
    position_update_numerator: typing.Union[None, npt.NDArray],
    position_update_denominator: typing.Union[None, npt.NDArray],
    batches: typing.List[npt.NDArray[np.intc]],
    streams: typing.List[cp.cuda.Stream],
    *,
    n: int,
    op: tike.operators.Ptycho,
    epoch: int,
) -> typing.Tuple[
    float,
    npt.ArrayLike,
    npt.ArrayLike,
    npt.ArrayLike,
    npt.ArrayLike,
    PtychoParameters,
]:
    cost = cp.zeros(1)
    count = cp.array(1.0 / len(batches[n]))
    psi_update_numerator = (
        cp.zeros_like(parameters.psi)
        if psi_update_numerator is None
        else psi_update_numerator
    )
    probe_update_numerator = (
        cp.zeros_like(parameters.probe)
        if probe_update_numerator is None
        else probe_update_numerator
    )
    position_update_numerator = (
        cp.empty_like(parameters.scan)
        if position_update_numerator is None
        else position_update_numerator
    )
    position_update_denominator = (
        cp.empty_like(parameters.scan)
        if position_update_denominator is None
        else position_update_denominator
    )

    def keep_some_args_constant(
        ind_args,
        lo: int,
        hi: int,
    ):
        (data,) = ind_args
        nonlocal cost, psi_update_numerator, probe_update_numerator
        nonlocal position_update_numerator, position_update_denominator

        unique_probe = tike.ptycho.probe.get_varying_probe(
            parameters.probe,
            parameters.eigen_probe,
            parameters.eigen_weights[lo:hi]
            if parameters.eigen_weights is not None
            else None,
        )

        farplane = op.fwd(
            probe=unique_probe,
            scan=parameters.scan[lo:hi],
            psi=parameters.psi,
        )
        intensity = cp.sum(
            cp.square(cp.abs(farplane)),
            axis=list(range(1, farplane.ndim - 2)),
        )
        each_cost = getattr(
            tike.operators,
            f"{parameters.exitwave_options.noise_model}_each_pattern",
        )(
            data[:, parameters.exitwave_options.measured_pixels][:, None, :],
            intensity[:, parameters.exitwave_options.measured_pixels][:, None, :],
        )
        cost += cp.sum(each_cost) * count

        if parameters.exitwave_options.noise_model == "poisson":
            xi = (1 - data / intensity)[:, None, None, :, :]
            grad_cost = farplane * xi

            step_length = cp.full(
                shape=(farplane.shape[0], 1, farplane.shape[2], 1, 1),
                fill_value=parameters.exitwave_options.step_length_start,
            )

            if parameters.exitwave_options.step_length_usemodes == "dominant_mode":
                step_length = tike.ptycho.exitwave.poisson_steplength_dominant_mode(
                    xi,
                    intensity,
                    data,
                    parameters.exitwave_options.measured_pixels,
                    step_length,
                    parameters.exitwave_options.step_length_weight,
                )

            else:

                step_length = tike.ptycho.exitwave.poisson_steplength_all_modes(
                    xi,
                    cp.square(cp.abs(farplane)),
                    intensity,
                    data,
                    parameters.exitwave_options.measured_pixels,
                    step_length,
                    parameters.exitwave_options.step_length_weight,
                )

            farplane[..., parameters.exitwave_options.measured_pixels] = (
                -step_length * grad_cost
            )[..., parameters.exitwave_options.measured_pixels]

        else:

            # Gaussian noise model for exitwave updates, steplength = 1
            # TODO: optimal step lengths using 2nd order taylor expansion

            farplane[..., parameters.exitwave_options.measured_pixels] = -getattr(
                tike.operators, f"{parameters.exitwave_options.noise_model}_grad"
            )(
                data,
                farplane,
                intensity,
            )[..., parameters.exitwave_options.measured_pixels]

        unmeasured_pixels = cp.logical_not(parameters.exitwave_options.measured_pixels)
        farplane[..., unmeasured_pixels] *= (
            parameters.exitwave_options.unmeasured_pixels_scaling - 1.0
        )

        pad, end = op.diffraction.pad, op.diffraction.end
        diff = op.propagation.adj(farplane, overwrite=True)[..., pad:end,
                                                            pad:end]

        if parameters.object_options:
            grad_psi = (
                cp.conj(unique_probe) * diff / parameters.probe.shape[-3]
            ).reshape(
                parameters.scan[lo:hi].shape[0] * parameters.probe.shape[-3],
                *parameters.probe.shape[-2:],
            )
            psi_update_numerator = op.diffraction.patch.adj(
                patches=grad_psi,
                images=psi_update_numerator,
                positions=parameters.scan[lo:hi],
                nrepeat=parameters.probe.shape[-3],
            )

        if parameters.position_options or parameters.probe_options:
            patches = op.diffraction.patch.fwd(
                patches=cp.zeros_like(diff[..., 0, 0, :, :]),
                images=parameters.psi,
                positions=parameters.scan[lo:hi],
            )[..., None, None, :, :]

        if parameters.probe_options and parameters.probe_options.update_start >= epoch:
            probe_update_numerator += cp.sum(
                cp.conj(patches) * diff,
                axis=-5,
                keepdims=True,
            )
            if parameters.eigen_weights:
                m: int = 0
                OP = patches * parameters.probe[..., m : m + 1, :, :]
                eigen_numerator = cp.sum(
                    cp.real(cp.conj(OP) * diff[..., m:m + 1, :, :]),
                    axis=(-1, -2),
                )
                eigen_denominator = cp.sum(
                    cp.abs(OP)**2,
                    axis=(-1, -2),
                )
                parameters.eigen_weights[lo:hi, ..., 0:1, m:m+1] += (
                    0.1 * (eigen_numerator / eigen_denominator)
                )  # yapf: disable

        if parameters.position_options:
            grad_x, grad_y = tike.ptycho.position.gaussian_gradient(patches)

            crop = parameters.probe.shape[-1] // 4

            position_update_numerator[lo:hi, ..., 0] = cp.sum(
                cp.real(
                    cp.conj(
                        grad_x[..., crop:-crop, crop:-crop]
                        * unique_probe[..., crop:-crop, crop:-crop]
                    )
                    * diff[..., crop:-crop, crop:-crop]
                ),
                axis=(-4, -3, -2, -1),
            )
            position_update_denominator[lo:hi, ..., 0] = cp.sum(
                cp.abs(
                    grad_x[..., crop:-crop, crop:-crop]
                    * unique_probe[..., crop:-crop, crop:-crop]
                )
                ** 2,
                axis=(-4, -3, -2, -1),
            )
            position_update_numerator[lo:hi, ..., 1] = cp.sum(
                cp.real(
                    cp.conj(
                        grad_y[..., crop:-crop, crop:-crop]
                        * unique_probe[..., crop:-crop, crop:-crop]
                    )
                    * diff[..., crop:-crop, crop:-crop]
                ),
                axis=(-4, -3, -2, -1),
            )
            position_update_denominator[lo:hi, ..., 1] = cp.sum(
                cp.abs(
                    grad_y[..., crop:-crop, crop:-crop]
                    * unique_probe[..., crop:-crop, crop:-crop]
                )
                ** 2,
                axis=(-4, -3, -2, -1),
            )

    tike.communicators.stream.stream_and_modify2(
        f=keep_some_args_constant,
        ind_args=[
            data,
        ],
        streams=streams,
        lo=batches[n][0],
        hi=batches[n][-1] + 1,
    )

    return (
        float(cost.get()),
        psi_update_numerator,
        probe_update_numerator,
        position_update_numerator,
        position_update_denominator,
        parameters,
    )


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
