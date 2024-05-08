import logging
import typing

import cupy as cp
import numpy.typing as npt

import tike.communicators
import tike.linalg
import tike.operators
import tike.opt
import tike.precision
import tike.ptycho.position
import tike.ptycho.probe
import tike.random

from .options import *

logger = logging.getLogger(__name__)


def dm(
    op: tike.operators.Ptycho,
    comm: tike.communicators.Comm,
    data: typing.List[npt.NDArray],
    batches: typing.List[typing.List[npt.NDArray[cp.intc]]],
    *,
    parameters: PtychoParameters,
    epoch: int,
) -> PtychoParameters:
    """Solve the ptychography problem using the difference map approach.

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
    Thibault, Pierre, Martin Dierolf, Oliver Bunk, Andreas Menzel, and Franz
    Pfeiffer. "Probe retrieval in ptychographic coherent diffractive imaging."
    Ultramicroscopy 109, no. 4 (2009): 338-343.

    .. seealso:: :py:mod:`tike.ptycho`

    """
    psi_update_numerator = [None] * comm.pool.num_workers
    probe_update_numerator = [None] * comm.pool.num_workers

    (
        cost,
        psi_update_numerator,
        probe_update_numerator,
    ) = (list(a) for a in zip(*comm.pool.map(
        _get_nearplane_gradients,
        data,
        parameters.scan,
        parameters.psi,
        parameters.probe,
        parameters.exitwave_options.measured_pixels,
        psi_update_numerator,
        probe_update_numerator,
        comm.streams,
        op=op,
        object_options=parameters.object_options,
        probe_options=parameters.probe_options,
        exitwave_options=parameters.exitwave_options,
    )))

    cost = comm.Allreduce_mean(cost).get()

    (
        parameters.psi,
        parameters.probe,
    ) = _apply_update(
        comm,
        psi_update_numerator,
        probe_update_numerator,
        parameters.psi,
        parameters.probe,
        parameters.object_options,
        parameters.probe_options,
    )

    parameters.algorithm_options.costs.append(cost)
    return parameters


def _apply_update(
    comm,
    psi_update_numerator,
    probe_update_numerator,
    psi,
    probe,
    object_options,
    probe_options,
):

    if object_options:
        psi_update_numerator = comm.Allreduce_reduce_gpu(
            psi_update_numerator)[0]

        new_psi = psi_update_numerator / (object_options.preconditioner[0] +
                                          1e-9)
        if object_options.use_adaptive_moment:
            (
                dpsi,
                object_options.v,
                object_options.m,
            ) = tike.opt.adam(
                g=(new_psi - psi[0]),
                v=object_options.v,
                m=object_options.m,
                vdecay=object_options.vdecay,
                mdecay=object_options.mdecay,
            )
            new_psi = dpsi + psi[0]
        psi = comm.pool.bcast([new_psi])

    if probe_options:

        probe_update_numerator = comm.Allreduce_reduce_gpu(
            probe_update_numerator)[0]

        new_probe = probe_update_numerator / (probe_options.preconditioner[0] +
                                              1e-9)
        if probe_options.use_adaptive_moment:
            (
                dprobe,
                probe_options.v,
                probe_options.m,
            ) = tike.opt.adam(
                g=(new_probe - probe[0]),
                v=probe_options.v,
                m=probe_options.m,
                vdecay=probe_options.vdecay,
                mdecay=probe_options.mdecay,
            )
            new_probe = dprobe + probe[0]
        probe = comm.pool.bcast([new_probe])

    return psi, probe


def _get_nearplane_gradients(
    data: npt.NDArray,
    scan: npt.NDArray,
    psi: npt.NDArray,
    probe: npt.NDArray,
    measured_pixels: npt.NDArray,
    psi_update_numerator: typing.Union[None, npt.NDArray],
    probe_update_numerator: typing.Union[None, npt.NDArray],
    streams: typing.List[cp.cuda.Stream],
    *,
    op: tike.operators.Ptycho,
    object_options: typing.Union[None, ObjectOptions] = None,
    probe_options: typing.Union[None, ProbeOptions] = None,
    exitwave_options: ExitWaveOptions,
) -> typing.List[npt.NDArray]:

    cost = cp.zeros(1, dtype=tike.precision.floating)
    count = cp.ones(1, dtype=tike.precision.floating) / len(data)
    probe_update_numerator = cp.zeros_like(
        probe) if probe_update_numerator is None else probe_update_numerator
    psi_update_numerator = cp.zeros_like(
        psi) if psi_update_numerator is None else psi_update_numerator

    def keep_some_args_constant(
        ind_args,
        lo,
        hi,
    ):
        (data,) = ind_args
        nonlocal cost, psi_update_numerator, probe_update_numerator

        varying_probe = probe

        farplane = op.fwd(probe=varying_probe, scan=scan[lo:hi], psi=psi)
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

        farplane[..., measured_pixels] *= ((
            cp.sqrt(data) / (cp.sqrt(intensity) + 1e-9))[..., None, None,
                                                         measured_pixels])
        farplane[..., ~measured_pixels] = 0

        pad, end = op.diffraction.pad, op.diffraction.end
        nearplane = op.propagation.adj(farplane, overwrite=True)[..., pad:end,
                                                                 pad:end]

        patches = op.diffraction.patch.fwd(
            patches=cp.zeros_like(nearplane[..., 0, 0, :, :]),
            images=psi,
            positions=scan[lo:hi],
        )[..., None, None, :, :]

        if object_options:

            grad_psi = (cp.conj(varying_probe) * nearplane).reshape(
                (hi - lo) * probe.shape[-3], *probe.shape[-2:])
            psi_update_numerator = op.diffraction.patch.adj(
                patches=grad_psi,
                images=psi_update_numerator,
                positions=scan[lo:hi],
                nrepeat=probe.shape[-3],
            )

        if probe_options:
            probe_update_numerator += cp.sum(
                cp.conj(patches) * nearplane,
                axis=-5,
                keepdims=True,
            )

    tike.communicators.stream.stream_and_modify2(
        f=keep_some_args_constant,
        ind_args=[
            data,
        ],
        streams=streams,
        lo=0,
        hi=len(data),
    )

    return [
        cost,
        psi_update_numerator,
        probe_update_numerator,
    ]
