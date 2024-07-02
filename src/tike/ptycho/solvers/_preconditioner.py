import typing

import cupy as cp
import numpy.typing as npt

import tike.communicators
import tike.operators
import tike.precision

from .options import ObjectOptions, ProbeOptions, PtychoParameters


def _rolling_average_object(parameters: PtychoParameters, new):
    if parameters.object_options.preconditioner is None:
        parameters.object_options.preconditioner = new
    else:
        parameters.object_options.preconditioner = 0.5 * (
            new + parameters.object_options.preconditioner
        )
    return parameters


def _rolling_average_probe(parameters: PtychoParameters, new):
    if parameters.probe_options.preconditioner is None:
        parameters.probe_options.preconditioner = new
    else:
        parameters.probe_options.preconditioner = 0.5 * (
            new + parameters.probe_options.preconditioner
        )
    return parameters


@cp.fuse()
def _probe_amp_sum(probe):
    return cp.sum(
        probe * cp.conj(probe),
        axis=-3,
    )


def _psi_preconditioner(
    parameters: PtychoParameters,
    streams: typing.List[cp.cuda.Stream],
    *,
    operator: tike.operators.Ptycho,
) -> npt.NDArray:

    # FIXME: Generated only one preconditioner for all slices
    psi_update_denominator = cp.zeros(
        shape=parameters.psi.shape[-2:],
        dtype=parameters.psi.dtype,
    )

    def make_certain_args_constant(
        ind_args,
        lo: int,
        hi: int,
    ) -> None:
        nonlocal psi_update_denominator

        probe_amp = _probe_amp_sum(parameters.probe)[:, 0]
        psi_update_denominator[...] = operator.diffraction.patch.adj(
            patches=probe_amp,
            images=psi_update_denominator,
            positions=parameters.scan[lo:hi],
        )

    tike.communicators.stream.stream_and_modify2(
        f=make_certain_args_constant,
        ind_args=[],
        streams=streams,
        lo=0,
        hi=len(parameters.scan),
    )

    return psi_update_denominator[None, ...]


@cp.fuse()
def _patch_amp_sum(patches):
    return cp.sum(
        patches * cp.conj(patches),
        axis=0,
        keepdims=False,
    )


def _probe_preconditioner(
    parameters: PtychoParameters,
    streams: typing.List[cp.cuda.Stream],
    *,
    operator: tike.operators.Ptycho,
) -> npt.NDArray:

    probe_update_denominator = cp.zeros(
        shape=parameters.probe.shape[-2:],
        dtype=parameters.probe.dtype,
    )

    def make_certain_args_constant(
        ind_args,
        lo: int,
        hi: int,
    ) -> None:
        nonlocal probe_update_denominator

        # FIXME: Only use the first slice for the probe preconditioner
        patches = operator.diffraction.patch.fwd(
            images=parameters.psi[0],
            positions=parameters.scan[lo:hi],
            patch_width=parameters.probe.shape[-1],
        )
        probe_update_denominator[...] += _patch_amp_sum(patches)
        assert probe_update_denominator.ndim == 2

    tike.communicators.stream.stream_and_modify2(
        f=make_certain_args_constant,
        ind_args=[],
        streams=streams,
        lo=0,
        hi=len(parameters.scan),
    )

    return probe_update_denominator


def update_preconditioners(
    comm: tike.communicators.Comm,
    parameters: typing.List[PtychoParameters],
    operator: tike.operators.Ptycho,
) -> typing.List[PtychoParameters]:
    """Update the probe and object preconditioners."""

    if parameters[0].object_options:
        preconditioner = comm.pool.map(
            _psi_preconditioner,
            parameters,
            comm.streams,
            operator=operator,
        )

        # preconditioner = comm.Allreduce(preconditioner)

        parameters = comm.pool.map(
            _rolling_average_object,
            parameters,
            preconditioner,
        )

    if parameters[0].probe_options:
        preconditioner = comm.pool.map(
            _probe_preconditioner,
            parameters,
            comm.streams,
            operator=operator,
        )

        # preconditioner = comm.Allreduce(preconditioner)

        parameters = comm.pool.map(
            _rolling_average_probe,
            parameters,
            preconditioner,
        )

    return parameters
