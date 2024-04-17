import logging

import cupy as cp
import cupyx.scipy.stats
import numpy.typing as npt

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

from .options import *
from .lstsq import _momentum_checked

logger = logging.getLogger(__name__)


def rpie(
    op: tike.operators.Ptycho,
    comm: tike.communicators.Comm,
    data: typing.List[npt.NDArray],
    batches: typing.List[typing.List[npt.NDArray[cp.intc]]],
    *,
    parameters: PtychoParameters,
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
    probe = parameters.probe
    scan = parameters.scan
    psi = parameters.psi
    algorithm_options = parameters.algorithm_options
    exitwave_options = parameters.exitwave_options
    probe_options = parameters.probe_options
    if probe_options is None:
        recover_probe = False
    else:
        recover_probe = probe_options.recover_probe

    position_options = parameters.position_options
    object_options = parameters.object_options
    eigen_probe = parameters.eigen_probe
    eigen_weights = parameters.eigen_weights

    if eigen_probe is None:
        beigen_probe = [None] * comm.pool.num_workers
    else:
        beigen_probe = eigen_probe

    if eigen_weights is None:
        beigen_weights = [None] * comm.pool.num_workers
    else:
        beigen_weights = eigen_weights

    if parameters.algorithm_options.batch_method == 'compact':
        order = range
    else:
        order = tike.random.randomizer_np.permutation

    psi_update_numerator = [None] * comm.pool.num_workers
    probe_update_numerator = [None] * comm.pool.num_workers
    position_update_numerator = [None] * comm.pool.num_workers
    position_update_denominator = [None] * comm.pool.num_workers

    batch_cost: typing.List[float] = []
    for n in order(algorithm_options.num_batch):

        (
            cost,
            psi_update_numerator,
            probe_update_numerator,
            position_update_numerator,
            position_update_denominator,
            beigen_weights,
        ) = (list(a) for a in zip(*comm.pool.map(
            _get_nearplane_gradients,
            data,
            scan,
            psi,
            probe,
            exitwave_options.measured_pixels,
            psi_update_numerator,
            probe_update_numerator,
            position_update_numerator,
            position_update_denominator,
            beigen_probe,
            beigen_weights,
            batches,
            comm.streams,
            n=n,
            op=op,
            object_options=object_options,
            probe_options=probe_options,
            recover_probe=recover_probe,
            position_options=position_options,
            exitwave_options=exitwave_options,
        )))

        batch_cost.append(comm.Allreduce_mean(cost, axis=None).get())

        if algorithm_options.batch_method != 'compact':
            (
                psi,
                probe,
            ) = _update(
                comm,
                psi,
                probe,
                psi_update_numerator,
                probe_update_numerator,
                object_options,
                probe_options,
                recover_probe,
                algorithm_options,
            )
            psi_update_numerator = [None] * comm.pool.num_workers
            probe_update_numerator = [None] * comm.pool.num_workers

    algorithm_options.costs.append(batch_cost)

    if position_options is not None:
        (
            scan,
            position_options,
        ) = (list(a) for a in zip(*comm.pool.map(
            _update_position,
            scan,
            position_options,
            position_update_numerator,
            position_update_denominator,
            max_shift=probe[0].shape[-1] * 0.1,
            alpha=algorithm_options.alpha,
            epoch=epoch,
        )))

    if algorithm_options.batch_method == 'compact':
        (
            psi,
            probe,
        ) = _update(
            comm,
            psi,
            probe,
            psi_update_numerator,
            probe_update_numerator,
            object_options,
            probe_options,
            recover_probe,
            algorithm_options,
            errors=list(np.mean(x) for x in algorithm_options.costs[-3:]),
        )

    if eigen_weights is not None:
        eigen_weights = comm.pool.map(
            _normalize_eigen_weights,
            beigen_weights,
        )

    parameters.probe = probe
    parameters.psi = psi
    parameters.scan = scan
    parameters.algorithm_options = algorithm_options
    parameters.probe_options = probe_options
    parameters.object_options = object_options
    parameters.position_options = position_options
    parameters.eigen_weights = eigen_weights
    return parameters


def _normalize_eigen_weights(eigen_weights):
    return eigen_weights / tike.linalg.mnorm(
        eigen_weights,
        axis=(-3),
        keepdims=True,
    )


def _update(
    comm: tike.communicators.Comm,
    psi: npt.NDArray[cp.csingle],
    probe: npt.NDArray[cp.csingle],
    psi_update_numerator: npt.NDArray[cp.csingle],
    probe_update_numerator: npt.NDArray[cp.csingle],
    object_options: ObjectOptions,
    probe_options: ProbeOptions,
    recover_probe: bool,
    algorithm_options: RpieOptions,
    errors: typing.Union[None, typing.List[float]] = None,
):
    if object_options:
        psi_update_numerator = comm.Allreduce_reduce_gpu(
            psi_update_numerator)[0]
        dpsi = psi_update_numerator
        deno = (
            (1 - algorithm_options.alpha) * object_options.preconditioner[0] +
            algorithm_options.alpha * object_options.preconditioner[0].max(
                axis=(-2, -1),
                keepdims=True,
            ))
        psi[0] = psi[0] + dpsi / deno
        if object_options.use_adaptive_moment:
            if errors:
                (
                    dpsi,
                    object_options.v,
                    object_options.m,
                ) = _momentum_checked(
                    g=dpsi,
                    v=object_options.v,
                    m=object_options.m,
                    mdecay=object_options.mdecay,
                    errors=errors,
                    memory_length=3,
                )
            else:
                (
                    dpsi,
                    object_options.v,
                    object_options.m,
                ) = tike.opt.adam(
                    g=dpsi,
                    v=object_options.v,
                    m=object_options.m,
                    vdecay=object_options.vdecay,
                    mdecay=object_options.mdecay,
                )
            psi[0] = psi[0] + dpsi / deno
        psi = comm.pool.bcast([psi[0]])

    if recover_probe:

        probe_update_numerator = comm.Allreduce_reduce_gpu(
            probe_update_numerator)[0]
        b0 = tike.ptycho.probe.finite_probe_support(
            probe[0],
            p=probe_options.probe_support,
            radius=probe_options.probe_support_radius,
            degree=probe_options.probe_support_degree,
        )
        b1 = probe_options.additional_probe_penalty * cp.linspace(
            0, 1, probe[0].shape[-3], dtype='float32')[..., None, None]
        dprobe = (probe_update_numerator - (b1 + b0) * probe[0])
        deno = (
            (1 - algorithm_options.alpha) * probe_options.preconditioner[0] +
            algorithm_options.alpha * probe_options.preconditioner[0].max(
                axis=(-2, -1),
                keepdims=True,
            ) + b0 + b1)
        probe[0] = probe[0] + dprobe / deno
        if probe_options.use_adaptive_moment:
            # ptychoshelves only applies momentum to the main probe
            mode = 0
            if errors:
                (
                    dprobe[0, 0, mode, :, :],
                    probe_options.v,
                    probe_options.m,
                ) = _momentum_checked(
                    g=(dprobe)[0, 0, mode, :, :],
                    v=probe_options.v,
                    m=probe_options.m,
                    mdecay=probe_options.mdecay,
                    errors=errors,
                    memory_length=3,
                )
            else:
                (
                    dprobe[0, 0, mode, :, :],
                    probe_options.v,
                    probe_options.m,
                ) = tike.opt.adam(
                    g=(dprobe)[0, 0, mode, :, :],
                    v=probe_options.v,
                    m=probe_options.m,
                    vdecay=probe_options.vdecay,
                    mdecay=probe_options.mdecay,
                )
            probe[0] = probe[0] + dprobe / deno
        probe = comm.pool.bcast([probe[0]])

    return psi, probe


def _get_nearplane_gradients(
    data: npt.NDArray,
    scan: npt.NDArray,
    psi: npt.NDArray,
    probe: npt.NDArray,
    measured_pixels: npt.NDArray,
    psi_update_numerator: typing.Union[None, npt.NDArray],
    probe_update_numerator: typing.Union[None, npt.NDArray],
    position_update_numerator: typing.Union[None, npt.NDArray],
    position_update_denominator: typing.Union[None, npt.NDArray],
    eigen_probe: typing.Union[None, npt.NDArray],
    eigen_weights: typing.Union[None, npt.NDArray],
    batches: typing.List[typing.List[int]],
    streams: typing.List[cp.cuda.Stream],
    *,
    n: int,
    op: tike.operators.Ptycho,
    object_options: typing.Union[None, ObjectOptions] = None,
    probe_options: typing.Union[None, ProbeOptions] = None,
    recover_probe: bool,
    position_options: typing.Union[None, PositionOptions],
    exitwave_options: ExitWaveOptions,
) -> typing.List[npt.NDArray]:

    cost = 0.0
    count = 1.0 / len(batches[n])
    psi_update_numerator = cp.zeros_like(
        psi) if psi_update_numerator is None else psi_update_numerator
    probe_update_numerator = cp.zeros_like(
        probe) if probe_update_numerator is None else probe_update_numerator
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
    ):
        (data,) = ind_args
        nonlocal cost, psi_update_numerator, probe_update_numerator
        nonlocal position_update_numerator, position_update_denominator
        nonlocal eigen_weights, scan

        unique_probe = tike.ptycho.probe.get_varying_probe(
            probe,
            eigen_probe,
            eigen_weights[lo:hi] if eigen_weights is not None else None,
        )

        farplane = op.fwd(probe=unique_probe, scan=scan[lo:hi], psi=psi)
        intensity = cp.sum(
            cp.square(cp.abs(farplane)),
            axis=list(range(1, farplane.ndim - 2)),
        )
        each_cost = getattr(
            tike.operators,
            f'{exitwave_options.noise_model}_each_pattern',
        )(
            data[:, measured_pixels][:, None, :],
            intensity[:, measured_pixels][:, None, :],
        )
        cost += cp.sum(each_cost) * count

        if exitwave_options.noise_model == 'poisson':

            xi = (1 - data / intensity)[:, None, None, :, :]
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

            farplane[..., measured_pixels] = (-step_length *
                                              grad_cost)[..., measured_pixels]

        else:

            # Gaussian noise model for exitwave updates, steplength = 1
            # TODO: optimal step lengths using 2nd order taylor expansion

            farplane[..., measured_pixels] = -getattr(
                tike.operators, f'{exitwave_options.noise_model}_grad')(
                    data,
                    farplane,
                    intensity,
                )[..., measured_pixels]

        unmeasured_pixels = cp.logical_not(measured_pixels)
        farplane[..., unmeasured_pixels] *= (
            exitwave_options.unmeasured_pixels_scaling - 1.0)

        pad, end = op.diffraction.pad, op.diffraction.end
        diff = op.propagation.adj(farplane, overwrite=True)[..., pad:end,
                                                            pad:end]

        if object_options:
            grad_psi = (cp.conj(unique_probe) * diff / probe.shape[-3]).reshape(
                scan[lo:hi].shape[0] * probe.shape[-3], *probe.shape[-2:])
            psi_update_numerator = op.diffraction.patch.adj(
                patches=grad_psi,
                images=psi_update_numerator,
                positions=scan[lo:hi],
                nrepeat=probe.shape[-3],
            )

        if position_options or probe_options:

            patches = op.diffraction.patch.fwd(
                patches=cp.zeros_like(diff[..., 0, 0, :, :]),
                images=psi,
                positions=scan[lo:hi],
            )[..., None, None, :, :]

        if recover_probe:
            probe_update_numerator += cp.sum(
                cp.conj(patches) * diff,
                axis=-5,
                keepdims=True,
            )
            if eigen_weights is not None:
                m: int = 0
                OP = patches * probe[..., m:m + 1, :, :]
                eigen_numerator = cp.sum(
                    cp.real(cp.conj(OP) * diff[..., m:m + 1, :, :]),
                    axis=(-1, -2),
                )
                eigen_denominator = cp.sum(
                    cp.abs(OP)**2,
                    axis=(-1, -2),
                )
                eigen_weights[lo:hi, ..., 0:1, m:m+1] += (
                    0.1 * (eigen_numerator / eigen_denominator)
                )  # yapf: disable

        if position_options:
            grad_x, grad_y = tike.ptycho.position.gaussian_gradient(patches)

            crop = probe.shape[-1] // 4

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
        cost,
        psi_update_numerator,
        probe_update_numerator,
        position_update_numerator,
        position_update_denominator,
        eigen_weights,
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
