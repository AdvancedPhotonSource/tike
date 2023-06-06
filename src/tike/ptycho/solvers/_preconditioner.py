import typing

import cupy as cp
import numpy.typing as npt

import tike.precision

from .options import ObjectOptions, ProbeOptions


@cp.fuse()
def _rolling_average(old, new):
    return 0.5 * (new + old)


def _psi_preconditioner(
    psi: npt.NDArray[tike.precision.cfloating],
    scan: npt.NDArray[tike.precision.floating],
    probe: npt.NDArray[tike.precision.cfloating],
    streams: typing.List[cp.cuda.Stream],
    *,
    operator: tike.operators.Ptycho,
) -> npt.NDArray:

    def make_certain_args_constant(
        scan: npt.NDArray,) -> typing.Tuple[npt.NDArray]:
        probe_amp = cp.sum(
            probe * probe.conj(),
            axis=-3,
        )[:, 0]
        psi_update_denominator = cp.zeros(
            shape=psi.shape,
            dtype=tike.precision.cfloating,
        )
        psi_update_denominator = operator.diffraction.patch.adj(
            patches=probe_amp,
            images=psi_update_denominator,
            positions=scan,
        )
        return (psi_update_denominator,)

    return tike.communicators.stream.stream_and_reduce(
        f=make_certain_args_constant,
        args=[
            scan,
        ],
        y_shapes=[
            psi.shape,
        ],
        y_dtypes=[
            psi.dtype,
        ],
        streams=streams,
    )[0]


def _probe_preconditioner(
    psi: npt.NDArray[tike.precision.cfloating],
    scan: npt.NDArray[tike.precision.floating],
    probe: npt.NDArray[tike.precision.cfloating],
    streams: typing.List[cp.cuda.Stream],
    *,
    operator: tike.operators.Ptycho,
) -> npt.NDArray:

    def make_certain_args_constant(
        scan: npt.NDArray,) -> typing.Tuple[npt.NDArray]:
        patches = operator.diffraction.patch.fwd(
            images=psi,
            positions=scan,
            patch_width=probe.shape[-1],
        )
        probe_update_denominator = cp.sum(
            patches * patches.conj(),
            axis=0,
            keepdims=False,
        )
        assert probe_update_denominator.ndim == 2
        return (probe_update_denominator,)

    return tike.communicators.stream.stream_and_reduce(
        f=make_certain_args_constant,
        args=[
            scan,
        ],
        y_shapes=[
            probe.shape[-2:],
        ],
        y_dtypes=[
            probe.dtype,
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
