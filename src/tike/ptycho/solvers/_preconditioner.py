import typing

import cupy as cp
import numpy.typing as npt

import tike.precision

from .options import ObjectOptions, ProbeOptions


@cp.fuse()
def _rolling_average(old, new):
    return 0.5 * (new + old)


@cp.fuse()
def _probe_amp_sum(probe):
    return cp.sum(
        probe * cp.conj(probe),
        axis=-3,
    )


def _psi_preconditioner(
    psi: npt.NDArray[tike.precision.cfloating],
    scan: npt.NDArray[tike.precision.floating],
    probe: npt.NDArray[tike.precision.cfloating],
    streams: typing.List[cp.cuda.Stream],
    *,
    operator: tike.operators.Ptycho,
) -> npt.NDArray:

    def make_certain_args_constant(
        ind_args,
        mod_args,
        _,
    ) -> typing.Tuple[npt.NDArray]:

        scan = ind_args[0]
        psi_update_denominator = mod_args[0]

        probe_amp = _probe_amp_sum(probe)[:, 0]
        psi_update_denominator = operator.diffraction.patch.adj(
            patches=probe_amp,
            images=psi_update_denominator,
            positions=scan,
        )
        return (psi_update_denominator,)

    psi_update_denominator = cp.zeros(
        shape=psi.shape,
        dtype=psi.dtype,
    )

    return tike.communicators.stream.stream_and_modify(
        f=make_certain_args_constant,
        ind_args=[
            scan,
        ],
        mod_args=[
            psi_update_denominator,
        ],
        streams=streams,
    )[0]


@cp.fuse()
def _patch_amp_sum(patches):
    return cp.sum(
        patches * cp.conj(patches),
        axis=0,
        keepdims=False,
    )


def _probe_preconditioner(
    psi: npt.NDArray[tike.precision.cfloating],
    scan: npt.NDArray[tike.precision.floating],
    probe: npt.NDArray[tike.precision.cfloating],
    streams: typing.List[cp.cuda.Stream],
    *,
    operator: tike.operators.Ptycho,
) -> npt.NDArray:

    def make_certain_args_constant(
        ind_args,
        mod_args,
        _,
    ) -> typing.Tuple[npt.NDArray]:
        scan = ind_args[0]
        probe_update_denominator = mod_args[0]

        patches = operator.diffraction.patch.fwd(
            images=psi,
            positions=scan,
            patch_width=probe.shape[-1],
        )
        probe_update_denominator += _patch_amp_sum(patches)
        assert probe_update_denominator.ndim == 2
        return (probe_update_denominator,)

    probe_update_denominator = cp.zeros(
        shape=probe.shape[-2:],
        dtype=probe.dtype,
    )

    return tike.communicators.stream.stream_and_modify(
        f=make_certain_args_constant,
        ind_args=[
            scan,
        ],
        mod_args=[
            probe_update_denominator,
        ],
        streams=streams,
    )[0]


def update_preconditioners(
    comm: tike.communicators.Comm,
    operator: tike.operators.Ptycho,
    scan,
    probe,
    psi,
    object_options: typing.Optional[ObjectOptions] = None,
    probe_options: typing.Optional[ProbeOptions] = None,
) -> typing.Tuple[ObjectOptions, ProbeOptions]:
    """Update the probe and object preconditioners."""
    if object_options:

        preconditioner = comm.pool.map(
            _psi_preconditioner,
            psi,
            scan,
            probe,
            comm.streams,
            operator=operator,
        )

        preconditioner = comm.Allreduce(preconditioner)

        if object_options.preconditioner is None:
            object_options.preconditioner = preconditioner
        else:
            object_options.preconditioner = comm.pool.map(
                _rolling_average,
                object_options.preconditioner,
                preconditioner,
            )

    if probe_options:

        preconditioner = comm.pool.map(
            _probe_preconditioner,
            psi,
            scan,
            probe,
            comm.streams,
            operator=operator,
        )

        preconditioner = comm.Allreduce(preconditioner)

        if probe_options.preconditioner is None:
            probe_options.preconditioner = preconditioner
        else:
            probe_options.preconditioner = comm.pool.map(
                _rolling_average,
                probe_options.preconditioner,
                preconditioner,
            )

    return object_options, probe_options
