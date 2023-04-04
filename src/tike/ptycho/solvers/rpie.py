import logging

import cupy as cp
import numpy.typing as npt

import tike.communicators
import tike.linalg
import tike.operators.cupy.objective as objective
import tike.opt
import tike.ptycho.object
import tike.ptycho.position
import tike.ptycho.probe
import tike.precision

from .options import *
from .lstsq import _momentum_checked

logger = logging.getLogger(__name__)


def rpie(
    op,
    comm,
    data,
    batches,
    *,
    parameters,
):
    """Solve the ptychography problem using regularized ptychographical engine.

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
    Maiden, Andrew M., and John M. Rodenburg. 2009. “An Improved
    Ptychographical Phase Retrieval Algorithm for Diffractive Imaging.”
    Ultramicroscopy 109 (10): 1256–62.
    https://doi.org/10.1016/j.ultramic.2009.05.012.

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
        for n in range(algorithm_options.num_batch):
            bscan = comm.pool.map(tike.opt.get_batch, scan, batches, n=n)
            preconditioner = comm.pool.map(
                _psi_preconditioner,
                preconditioner,
                probe,
                bscan,
                psi,
                op=op,
            )
        preconditioner = comm.Allreduce(preconditioner)
        if object_options.preconditioner is None:
            object_options.preconditioner = preconditioner
        else:
            object_options.preconditioner = comm.pool.map(
                _rolling_preconditioner,
                object_options.preconditioner,
                preconditioner,
            )

    if probe_options is not None:
        preconditioner = [None] * comm.pool.num_workers
        for n in range(algorithm_options.num_batch):
            bscan = comm.pool.map(tike.opt.get_batch, scan, batches, n=n)
            preconditioner = comm.pool.map(
                _probe_preconditioner,
                preconditioner,
                probe,
                bscan,
                psi,
                op=op,
            )
        preconditioner = comm.Allreduce(preconditioner)
        if probe_options.preconditioner is None:
            probe_options.preconditioner = preconditioner
        else:
            probe_options.preconditioner = comm.pool.map(
                _rolling_preconditioner,
                probe_options.preconditioner,
                preconditioner,
            )

    if parameters.algorithm_options.batch_method == 'compact':
        order = range
    else:
        order = tike.opt.randomizer.permutation

    batch_cost = []
    object_update_sum = 0
    probe_update_sum = 0
    for n in order(algorithm_options.num_batch):

        bdata = comm.pool.map(tike.opt.get_batch, data, batches, n=n)
        bscan = comm.pool.map(tike.opt.get_batch, scan, batches, n=n)

        if position_options is None:
            bposition_options = [None for b in batches]
        else:
            bposition_options = comm.pool.map(
                tike.ptycho.position.PositionOptions.split,
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

        (
            nearplane,
            cost,
            bposition_options,
        ) = zip(*comm.pool.map(
            _update_wavefront,
            bdata,
            unique_probe,
            bscan,
            psi,
            bposition_options,
            op=op,
        ))

        batch_cost.append(comm.Allreduce_mean(cost, axis=None).get())

        (
            psi_update_numerator,
            probe_update_numerator,
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
            position_options=bposition_options,
            algorithm_options=algorithm_options,
            probe_options=probe_options,
            object_options=object_options,
        )

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
                algorithm_options,
            )
        else:
            object_update_sum += psi_update_numerator
            probe_update_sum += probe_update_numerator

        if position_options is not None:
            comm.pool.map(
                tike.ptycho.position.PositionOptions.insert,
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

    if algorithm_options.batch_method == 'compact':
        (
            psi,
            probe,
        ) = _update(
            comm,
            psi,
            probe,
            object_update_sum,
            probe_update_sum,
            object_options,
            probe_options,
            algorithm_options,
            errors=list(np.mean(x) for x in algorithm_options.costs[-3:]),
        )

    if eigen_weights is not None:
        eigen_weights = comm.pool.map(
            _normalize_eigen_weights,
            eigen_weights,
        )

    if object_options:
        psi = comm.pool.map(tike.ptycho.object.positivity_constraint,
                            psi,
                            r=object_options.positivity_constraint)

        psi = comm.pool.map(tike.ptycho.object.smoothness_constraint,
                            psi,
                            a=object_options.smoothness_constraint)

    if object_options is not None and (len(algorithm_options.costs) % 10 == 1):
        (
            psi,
            probe,
        ) = (list(a) for a in zip(*comm.pool.map(
            tike.ptycho.object.remove_object_ambiguity,
            psi,
            probe,
            object_options.preconditioner,
        )))

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
    algorithm_options: RpieOptions,
    errors: typing.Union[None, typing.List[float]] = None,
):
    if object_options:
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

    if probe_options:
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
                    vdecay=object_options.vdecay,
                    mdecay=object_options.mdecay,
                )
            probe[0] = probe[0] + dprobe / deno
        probe = comm.pool.bcast([probe[0]])

    return psi, probe


@cp.fuse()
def _rolling_preconditioner(old, new):
    return 0.5 * (new + old)


def _update_wavefront(
    data,
    varying_probe,
    scan,
    psi,
    position_options,
    op=None,
):
    farplane = op.fwd(probe=varying_probe, scan=scan, psi=psi)
    intensity = cp.sum(
        cp.square(cp.abs(farplane)),
        axis=list(range(1, farplane.ndim - 2)),
    )
    cost = getattr(objective, f'{op.propagation.model}_each_pattern')(
        data,
        intensity,
    )
    if position_options is not None:
        position_options.confidence[..., 0] = cost
    cost = cp.mean(cost)
    logger.info('%10s cost is %+12.5e', 'farplane', cost)

    farplane *= (cp.sqrt(data) / (cp.sqrt(intensity) + 1e-9))[..., None,
                                                              None, :, :]
    farplane = op.propagation.adj(farplane, overwrite=True)

    pad, end = op.diffraction.pad, op.diffraction.end
    return farplane[..., pad:end, pad:end], cost, position_options


def _psi_preconditioner(
    psi_update_denominator,
    unique_probe,
    scan_,
    psi,
    *,
    op,
    m=0,
):
    # Sum of the probe amplitude over field of view for preconditioning the
    # object update.
    probe_amp = cp.sum(
        (unique_probe[..., 0, :, :, :] * unique_probe[..., 0, :, :, :].conj()),
        axis=-3,
    )
    if psi_update_denominator is None:
        psi_update_denominator = cp.zeros(
            shape=psi.shape,
            dtype=cp.csingle,
        )
    psi_update_denominator = op.diffraction.patch.adj(
        patches=probe_amp,
        images=psi_update_denominator,
        positions=scan_,
    )
    return psi_update_denominator


def _probe_preconditioner(
    probe_update_denominator,
    probe,
    scan_,
    psi,
    *,
    op,
):
    patches = op.diffraction.patch.fwd(
        images=psi,
        positions=scan_,
        patch_width=probe.shape[-1],
    )
    if probe_update_denominator is None:
        probe_update_denominator = 0
    probe_update_denominator += cp.sum(
        patches * patches.conj(),
        axis=0,
        keepdims=False,
    )
    assert probe_update_denominator.ndim == 2
    return probe_update_denominator


def _update_nearplane(
    op,
    comm,
    nearplane_,
    psi,
    scan_,
    probe,
    unique_probe,
    eigen_probe,
    eigen_weights,
    recover_psi,
    recover_probe,
    step_length=1.0,
    algorithm_options=None,
    position_options=[None],
    *,
    probe_options=None,
    object_options=None,
):

    patches = comm.pool.map(_get_patches, nearplane_, psi, scan_, op=op)

    (
        psi_update_numerator,
        _,
        probe_update_numerator,
        _,
        position_update_numerator,
        position_update_denominator,
        eigen_weights,
    ) = (list(a) for a in zip(*comm.pool.map(
        _get_nearplane_gradients,
        nearplane_,
        patches,
        psi,
        scan_,
        probe,
        unique_probe,
        eigen_weights,
        recover_psi=recover_psi,
        recover_probe=recover_probe,
        recover_positions=position_options[0] is not None,
        op=op,
    )))

    alpha = algorithm_options.alpha

    if recover_psi:
        psi_update_numerator = comm.Allreduce_reduce_gpu(
            psi_update_numerator)[0]

    if recover_probe:
        probe_update_numerator = comm.Allreduce_reduce_gpu(
            probe_update_numerator)[0]

    if position_options[0] is not None:
        (
            scan_,
            position_options,
        ) = (list(a) for a in zip(*comm.pool.map(
            _update_position,
            scan_,
            position_options,
            position_update_numerator,
            position_update_denominator,
            max_shift=probe[0].shape[-1] * 0.1,
            alpha=alpha,
        )))

    return (
        psi_update_numerator,
        probe_update_numerator,
        eigen_probe,
        eigen_weights,
        scan_,
        position_options,
    )


def _get_patches(nearplane, psi, scan, op=None):
    patches = op.diffraction.patch.fwd(
        patches=cp.zeros_like(nearplane[..., 0, 0, :, :]),
        images=psi,
        positions=scan,
    )[..., None, None, :, :]
    return patches


def _get_nearplane_gradients(
    nearplane,
    patches,
    psi,
    scan,
    probe,
    unique_probe,
    eigen_weights,
    recover_psi=True,
    recover_probe=True,
    recover_positions=True,
    op=None,
):
    psi_update_numerator = cp.zeros_like(psi)
    probe_update_numerator = cp.zeros_like(probe)
    position_update_numerator = cp.zeros_like(scan)
    position_update_denominator = cp.zeros_like(scan)

    grad_x, grad_y = tike.ptycho.position.gaussian_gradient(patches)

    for m in range(probe.shape[-3]):

        diff = (nearplane[..., [m], :, :] -
                (unique_probe[..., [m], :, :] * patches))

        if recover_psi:
            grad_psi = (cp.conj(unique_probe[..., [m], :, :]) * diff /
                        probe.shape[-3])
            psi_update_numerator = op.diffraction.patch.adj(
                patches=grad_psi[..., 0, 0, :, :],
                images=psi_update_numerator,
                positions=scan,
            )

        if recover_probe:
            probe_update_numerator[..., [m], :, :] = cp.sum(
                cp.conj(patches) * diff,
                axis=-5,
                keepdims=True,
            )
            if m == 0 and eigen_weights is not None:
                OP = patches * probe[..., [m], :, :]
                eigen_numerator = cp.sum(
                    cp.real(cp.conj(OP) * diff[..., [m], :, :]),
                    axis=(-1, -2),
                )
                eigen_denominator = cp.sum(
                    cp.abs(OP)**2,
                    axis=(-1, -2),
                )
                eigen_weights[..., 0:1, [m]] += 0.1 * (eigen_numerator /
                                                       eigen_denominator)

        if recover_positions:
            position_update_numerator[..., 0] += cp.sum(
                cp.real(cp.conj(grad_x * unique_probe[..., [m], :, :]) * diff),
                axis=(-2, -1),
            )[..., 0, 0]
            position_update_denominator[..., 0] += cp.sum(
                cp.abs(grad_x * unique_probe[..., [m], :, :])**2,
                axis=(-2, -1),
            )[..., 0, 0]
            position_update_numerator[..., 1] += cp.sum(
                cp.real(cp.conj(grad_y * unique_probe[..., [m], :, :]) * diff),
                axis=(-2, -1),
            )[..., 0, 0]
            position_update_denominator[..., 1] += cp.sum(
                cp.abs(grad_y * unique_probe[..., [m], :, :])**2,
                axis=(-2, -1),
            )[..., 0, 0]

    return (
        psi_update_numerator,
        None,
        probe_update_numerator,
        None,
        position_update_numerator,
        position_update_denominator,
        eigen_weights,
    )


def _update_position(
    scan,
    position_options,
    position_update_numerator,
    position_update_denominator,
    alpha=0.05,
    max_shift=1,
):
    step = (position_update_numerator) / (
        (1 - alpha) * position_update_denominator +
        alpha * max(position_update_denominator.max(), 1e-6))

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
