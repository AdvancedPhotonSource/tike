import logging

import cupy as cp

import tike.linalg
import tike.opt
import tike.ptycho.position
import tike.ptycho.probe

from ..object import positivity_constraint, smoothness_constraint

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
    batch_cost = []

    for n in tike.opt.randomizer.permutation(len(batches[0])):

        bdata = comm.pool.map(tike.opt.get_batch, data, batches, n=n)
        bscan = comm.pool.map(tike.opt.get_batch, scan, batches, n=n)

        if position_options is None:
            bposition_options = None
        else:
            bposition_options = comm.pool.map(
                tike.ptycho.position.PositionOptions.split,
                position_options,
                [b[n] for b in batches],
            )

        unique_probe = probe
        beigen_probe = None
        beigen_weights = None

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
            position_options=bposition_options,
            algorithm_options=algorithm_options,
            probe_options=probe_options,
        )

        if position_options is not None:
            comm.pool.map(
                tike.ptycho.position.PositionOptions.insert,
                position_options,
                bposition_options,
                [b[n] for b in batches],
            )

        comm.pool.map(
            tike.opt.put_batch,
            bscan,
            scan,
            batches,
            n=n,
        )

    if probe_options and probe_options.orthogonality_constraint:
        probe = comm.pool.map(tike.ptycho.probe.orthogonalize_eig, probe)

    if object_options:
        psi = comm.pool.map(positivity_constraint,
                            psi,
                            r=object_options.positivity_constraint)

        psi = comm.pool.map(smoothness_constraint,
                            psi,
                            a=object_options.smoothness_constraint)

    algorithm_options.costs.append(batch_cost)
    parameters.probe = probe
    parameters.psi = psi
    parameters.scan = scan
    parameters.algorithm_options = algorithm_options
    parameters.probe_options = probe_options
    parameters.object_options = object_options
    parameters.position_options = position_options
    return parameters


def _update_wavefront(data, varying_probe, scan, psi, op=None):

    farplane = op.fwd(probe=varying_probe, scan=scan, psi=psi)
    intensity = cp.sum(
        cp.square(cp.abs(farplane)),
        axis=list(range(1, farplane.ndim - 2)),
    )
    cost = op.propagation.cost(data, intensity)
    logger.info('%10s cost is %+12.5e', 'farplane', cost)

    farplane *= (cp.sqrt(data) / (cp.sqrt(intensity) + 1e-9))[..., None,
                                                              None, :, :]

    farplane = op.propagation.adj(farplane, overwrite=True)

    pad, end = op.diffraction.pad, op.diffraction.end
    return farplane[..., pad:end, pad:end], cost


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
    position_options=None,
    *,
    probe_options=None,
):

    patches = comm.pool.map(_get_patches, nearplane_, psi, scan_, op=op)

    (
        psi_update_numerator,
        psi_update_denominator,
        probe_update_numerator,
        probe_update_denominator,
        position_update_numerator,
        position_update_denominator,
    ) = (list(a) for a in zip(*comm.pool.map(
        _get_nearplane_gradients,
        nearplane_,
        patches,
        psi,
        scan_,
        probe,
        recover_psi=recover_psi,
        recover_probe=recover_probe,
        recover_positions=position_options is not None,
        op=op,
    )))

    alpha = algorithm_options.alpha

    if recover_psi:
        if comm.use_mpi:
            psi_update_numerator = comm.Allreduce_reduce(
                psi_update_numerator, 'gpu')[0]
            psi_update_denominator = comm.Allreduce_reduce(
                psi_update_denominator, 'gpu')[0]
        else:
            psi_update_numerator = comm.reduce(psi_update_numerator, 'gpu')[0]
            psi_update_denominator = comm.reduce(psi_update_denominator,
                                                 'gpu')[0]

        psi[0] += step_length * psi_update_numerator / (
            (1 - alpha) * psi_update_denominator +
            alpha * psi_update_denominator.max(
                axis=(-2, -1),
                keepdims=True,
            ))

        psi = comm.pool.bcast([psi[0]])

    if recover_probe:
        if comm.use_mpi:
            probe_update_numerator = comm.Allreduce_reduce(
                probe_update_numerator, 'gpu')[0]
            probe_update_denominator = comm.Allreduce_reduce(
                probe_update_denominator, 'gpu')[0]
        else:
            probe_update_numerator = comm.reduce(probe_update_numerator,
                                                 'gpu')[0]
            probe_update_denominator = comm.reduce(probe_update_denominator,
                                                   'gpu')[0]

        b = tike.ptycho.probe.finite_probe_support(
            probe[0],
            p=probe_options.probe_support,
            radius=probe_options.probe_support_radius,
            degree=probe_options.probe_support_degree,
        )

        probe[0] += step_length * (probe_update_numerator - b * probe[0]) / (
            (1 - alpha) * probe_update_denominator +
            alpha * probe_update_denominator.max(
                axis=(-2, -1),
                keepdims=True,
            ) + b)

        probe = comm.pool.bcast([probe[0]])

    if position_options:
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

    return psi, probe, eigen_probe, eigen_weights, scan_, position_options


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


def _get_nearplane_gradients(
    nearplane,
    patches,
    psi,
    scan,
    probe,
    recover_psi=True,
    recover_probe=True,
    recover_positions=True,
    op=None,
):
    psi_update_numerator = cp.zeros(psi.shape, dtype='complex64')
    psi_update_denominator = cp.zeros(psi.shape, dtype='complex64')
    probe_update_numerator = cp.zeros(probe.shape, dtype='complex64')
    position_update_numerator = cp.zeros(scan.shape, dtype='float32')
    position_update_denominator = cp.zeros(scan.shape, dtype='float32')

    grad_x, grad_y = tike.ptycho.position._image_grad(op, patches)

    for m in range(probe.shape[-3]):

        diff = nearplane[..., [m], :, :] - (probe[..., [m], :, :] * patches)

        if recover_psi:
            grad_psi = cp.conj(probe[..., [m], :, :]) * diff
            psi_update_numerator = op.diffraction.patch.adj(
                patches=grad_psi[..., 0, 0, :, :],
                images=psi_update_numerator,
                positions=scan,
            )
            probe_amp = probe[..., 0, m, :, :] * probe[..., 0, m, :, :].conj()
            # TODO: Allow this kind of broadcasting inside the patch operator
            if probe_amp.shape[-3] == 1:
                probe_amp = cp.tile(probe_amp, (scan.shape[-2], 1, 1))
            psi_update_denominator = op.diffraction.patch.adj(
                patches=probe_amp,
                images=psi_update_denominator,
                positions=scan,
            )

        if recover_probe:
            probe_update_numerator[..., [m], :, :] = cp.sum(
                cp.conj(patches) * diff,
                axis=-5,
                keepdims=True,
            )

        if recover_positions:
            position_update_numerator[..., 0] += cp.sum(
                cp.real(cp.conj(grad_x * probe[..., [m], :, :]) * diff),
                axis=(-2, -1),
            )[..., 0, 0]
            position_update_denominator[..., 0] += cp.sum(
                cp.abs(grad_x * probe[..., [m], :, :])**2,
                axis=(-2, -1),
            )[..., 0, 0]
            position_update_numerator[..., 1] += cp.sum(
                cp.real(cp.conj(grad_y * probe[..., [m], :, :]) * diff),
                axis=(-2, -1),
            )[..., 0, 0]
            position_update_denominator[..., 1] += cp.sum(
                cp.abs(grad_y * probe[..., [m], :, :])**2,
                axis=(-2, -1),
            )[..., 0, 0]

    if recover_probe:
        probe_update_denominator = cp.sum(
            patches * patches.conj(),
            axis=-5,
            keepdims=True,
        )
    else:
        probe_update_denominator = None

    return (
        psi_update_numerator,
        psi_update_denominator,
        probe_update_numerator,
        probe_update_denominator,
        position_update_numerator,
        position_update_denominator,
    )


def _mad(x, **kwargs):
    """Return the mean absolute deviation around the median."""
    return cp.mean(cp.abs(x - cp.median(x, **kwargs)), **kwargs)


def _update_position(
    scan,
    position_options,
    position_update_numerator,
    position_update_denominator,
    alpha=0.05,
    max_shift=1,
):
    step = position_update_numerator / (
        (1 - alpha) * position_update_denominator +
        alpha * max(position_update_denominator.max(), 1e-6))

    step_x = step[..., 0]
    step_y = step[..., 1]

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

    # Ensure net movement is zero
    step_x -= cp.mean(step_x, axis=-1, keepdims=True)
    step_y -= cp.mean(step_y, axis=-1, keepdims=True)
    logger.info('position update norm is %+.3e', tike.linalg.norm(step_x))

    scan[..., 0] -= step_x
    scan[..., 1] -= step_y

    return scan, position_options
