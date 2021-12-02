import logging

import cupy as cp

from tike.linalg import lstsq, projection, norm, orthogonalize_gs
from tike.opt import get_batch, put_batch, randomizer

from ..object import positivity_constraint, smoothness_constraint
from ..probe import orthogonalize_eig

logger = logging.getLogger(__name__)


def epie(
    op,
    comm,
    data,
    probe,
    scan,
    psi,
    batches,
    probe_options=None,
    position_options=None,
    object_options=None,
    cost=None,
):
    """Solve the ptychography problem using extended ptychographical engine.

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
    batches : list(list((BATCH_SIZE, ) int, ...), ...)
        A list of list of indices along the FRAME axis of `data` for
        each device which define the batches of `data` to process
        simultaneously.
    position_options : :py:class:`tike.ptycho.PositionOptions`
        A class containing settings related to position correction.
    probe_options : :py:class:`tike.ptycho.ProbeOptions`
        A class containing settings related to probe updates.
    object_options : :py:class:`tike.ptycho.ObjectOptions`
        A class containing settings related to object updates.
    cost : float
        The current objective function value.

    Returns
    -------
    result : dict
        A dictionary containing the updated inputs if they can be updated.

    References
    ----------
    Maiden, Andrew M., and John M. Rodenburg. 2009. “An Improved
    Ptychographical Phase Retrieval Algorithm for Diffractive Imaging.”
    Ultramicroscopy 109 (10): 1256–62.
    https://doi.org/10.1016/j.ultramic.2009.05.012.

    .. seealso:: :py:mod:`tike.ptycho`

    """
    for n in randomizer.permutation(len(batches[0])):

        bdata = comm.pool.map(get_batch, data, batches, n=n)
        bscan = comm.pool.map(get_batch, scan, batches, n=n)

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
            cost = comm.Allreduce_reduce(cost, 'cpu')
        else:
            cost = comm.reduce(cost, 'cpu')

        (
            psi,
            probe,
            beigen_probe,
            beigen_weights,
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
        )

        comm.pool.map(
            put_batch,
            bscan,
            scan,
            batches,
            n=n,
        )

    if probe_options and probe_options.orthogonality_constraint:
        probe = comm.pool.map(orthogonalize_eig, probe)

    if object_options:
        psi = comm.pool.map(positivity_constraint,
                            psi,
                            r=object_options.positivity_constraint)

        psi = comm.pool.map(smoothness_constraint,
                            psi,
                            a=object_options.smoothness_constraint)

    result = {
        'psi': psi,
        'probe': probe,
        'cost': cost,
        'scan': scan,
        'probe_options': probe_options,
        'object_options': object_options,
        'position_options': position_options,
    }
    return result


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


def max_amplitude(x, **kwargs):
    """Return the maximum of the absolute square."""
    return (x * x.conj()).real.max(**kwargs)


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
):

    patches = comm.pool.map(_get_patches, nearplane_, psi, scan_, op=op)
    probe_step = step_length / scan_[0].shape[0]
    psi_step = probe_step / probe[0].shape[-3]

    for m in range(probe[0].shape[-3]):

        (
            common_grad_psi,
            common_grad_probe,
        ) = (list(a) for a in zip(*comm.pool.map(
            _get_nearplane_gradients,
            nearplane_,
            patches,
            psi,
            scan_,
            probe,
            m=m,
            recover_psi=recover_psi,
            recover_probe=recover_probe,
            op=op,
        )))

        if recover_psi:
            if comm.use_mpi:
                common_grad_psi = comm.Allreduce_reduce(
                    common_grad_psi,
                    'gpu',
                )[0]
            else:
                common_grad_psi = comm.reduce(common_grad_psi, 'gpu')[0]
            psi[0] += psi_step * common_grad_psi
            psi = comm.pool.bcast([psi[0]])

        if recover_probe:
            if comm.use_mpi:
                common_grad_probe = comm.Allreduce_reduce(
                    common_grad_probe,
                    'gpu',
                )[0]
            else:
                common_grad_probe = comm.reduce(common_grad_probe, 'gpu')[0]
            probe[0][..., [m], :, :] += probe_step * common_grad_probe
            probe = comm.pool.bcast([probe[0]])

    return psi, probe, eigen_probe, eigen_weights


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
    m=0,
    recover_psi=True,
    recover_probe=True,
    op=None,
):
    diff = nearplane[..., [m], :, :] - (probe[..., [m], :, :] * patches)

    if recover_psi:
        grad_psi = cp.conj(probe[..., [m], :, :]) * diff / max_amplitude(
            probe[..., [m], :, :],
            keepdims=True,
            axis=(-1, -2),
        )
        common_grad_psi = op.diffraction.patch.adj(
            patches=grad_psi[..., 0, 0, :, :],
            images=cp.zeros(psi.shape, dtype='complex64'),
            positions=scan,
        )

    if recover_probe:
        grad_probe = cp.conj(patches) * diff / max_amplitude(
            patches,
            keepdims=True,
            axis=(-1, -2),
        )
        common_grad_probe = cp.sum(
            grad_probe,
            axis=-5,
            keepdims=True,
        )

    return common_grad_psi, common_grad_probe
