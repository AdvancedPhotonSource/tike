import logging

import cupy as cp

import tike.linalg
import tike.opt
import tike.ptycho.position
import tike.ptycho.probe

from tike.ptycho.solvers.rpie import _update_position

logger = logging.getLogger(__name__)


def dm(
    op,
    comm,
    data,
    batches,
    *,
    parameters,
):
    """Solve the ptychography problem using the difference map approach.

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
    parameters : :py:class:`tike.ptycho.solvers.PtychoParameters`
        An object which contains reconstruction parameters.

    Returns
    -------
    result : :py:class:`tike.ptycho.solvers.PtychoParameters`
        An object which contains reconstruction parameters.

    References
    ----------

    .. seealso:: :py:mod:`tike.ptycho`

    """
    for n in tike.opt.randomizer.permutation(len(batches[0])):

        bdata = comm.pool.map(tike.opt.get_batch, data, batches, n=n)
        bscan = comm.pool.map(tike.opt.get_batch, parameters.scan, batches, n=n)

        if parameters.position_options is None:
            bposition_options = None
        else:
            bposition_options = comm.pool.map(
                tike.ptycho.position.PositionOptions.split,
                parameters.position_options,
                [b[n] for b in batches],
            )

        unique_probe = parameters.probe
        beigen_probe = None
        beigen_weights = None

        nearplane, cost = zip(*comm.pool.map(
            _update_wavefront,
            bdata,
            unique_probe,
            bscan,
            parameters.psi,
            op=op,
        ))

        if comm.use_mpi:
            # TODO: This reduction should be mean
            cost = comm.Allreduce_reduce(cost, 'cpu')
        else:
            cost = comm.reduce(cost, 'cpu')

        (
            parameters.psi,
            parameters.probe,
            beigen_probe,
            beigen_weights,
            bscan,
            bposition_options,
        ) = _update_nearplane(
            op,
            comm,
            nearplane,
            parameters.psi,
            bscan,
            parameters.probe,
            unique_probe,
            beigen_probe,
            beigen_weights,
            parameters.object_options is not None,
            parameters.probe_options is not None,
            position_options=bposition_options,
            algorithm_options=parameters.algorithm_options,
            probe_options=parameters.probe_options,
        )

        if parameters.position_options is not None:
            comm.pool.map(
                tike.ptycho.position.PositionOptions.insert,
                parameters.position_options,
                bposition_options,
                [b[n] for b in batches],
            )

        comm.pool.map(
            tike.opt.put_batch,
            bscan,
            parameters.scan,
            batches,
            n=n,
        )

    parameters.algorithm_options.costs.append(cost)
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
        # position_update_numerator,
        # position_update_denominator,
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

        psi[0] = psi_update_numerator / (psi_update_denominator + 1e-9)
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

        probe[0] = probe_update_numerator / (probe_update_denominator + 1e-9)
        probe = comm.pool.bcast([probe[0]])

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
    # position_update_numerator = cp.zeros(scan.shape, dtype='float32')
    # position_update_denominator = cp.zeros(scan.shape, dtype='float32')

    # grad_x, grad_y = tike.ptycho.position._image_grad(op, patches)

    for m in range(probe.shape[-3]):

        diff = nearplane[..., [m], :, :]

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
        # position_update_numerator,
        # position_update_denominator,
    )
