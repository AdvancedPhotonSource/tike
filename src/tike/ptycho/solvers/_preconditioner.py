import typing

import cupy as cp
import numpy.typing as npt

import tike.communicators
import tike.operators
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

    psi_update_denominator = cp.zeros(
        shape=psi.shape,
        dtype=psi.dtype,
    )

    def make_certain_args_constant(
        ind_args,
        lo: int,
        hi: int,
    ) -> None:
        
        nonlocal psi_update_denominator

        # probe0 = cp.repeat( probe, scan[lo:hi].shape[0], axis = 0)[..., 0, :, :, :]
        # probe_amp = _probe_amp_sum(probe0)

        #probe_amp = _probe_amp_sum(probe)[:, 0]         # _probe_amp_sum(probe)[ :, 0 ] = _probe_amp_sum(probe)[ :, 0, ... ]

        _, multislice_probes = operator.fwd( 
            probe,
            scan[lo:hi],
            psi,
        ) 

        for tt in cp.arange( 0, psi.shape[0], 1 ) :

            probe_amp = _probe_amp_sum( multislice_probes[ tt, ... ])    

            psi_update_denominator[ tt, ... ] = operator.multislice.diffraction.patch.adj(
                patches=probe_amp,
                images=psi_update_denominator[ tt, ... ],
                positions=scan[lo:hi],
            )

    tike.communicators.stream.stream_and_modify2(       # stream over scan positions
        f=make_certain_args_constant,
        ind_args=[],
        streams=streams,
        lo=0,
        hi=len(scan),
    )

    ''' 

    import matplotlib.pyplot as plt
    #import numpy as np
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib as mpl
    # mpl.use('Agg')
    mpl.use('TKAgg')

    A = cp.abs( psi_update_denominator[ 0, ... ] - psi_update_denominator[ 1, ... ])
    fig, ax1 = plt.subplots( nrows = 1, ncols = 1, )
    pos1 = ax1.imshow( A.get(), cmap = 'gray', ) 
    plt.colorbar(pos1)
    plt.show( block = False )
    
    '''


    return psi_update_denominator


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

    probe_update_denominator = cp.zeros(
        shape=( psi.shape[0], *probe.shape[-2:] ),
        dtype=probe.dtype,
    )

    def make_certain_args_constant(
        ind_args,
        lo: int,
        hi: int,
    ) -> None:
        
        nonlocal probe_update_denominator

        # patches = cp.zeros( )
        
        for tt in cp.arange( 0, psi.shape[0], 1 ) :

            patches = operator.multislice.diffraction.patch.fwd(
                images=psi[ tt, ... ],
                positions=scan[lo:hi],
                patch_width=probe.shape[-1],
            )

            probe_update_denominator[tt, ...] += _patch_amp_sum(patches)

        assert probe_update_denominator.ndim == 3

        ''' 

        import matplotlib.pyplot as plt
        #import numpy as np
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib as mpl
        # mpl.use('Agg')
        mpl.use('TKAgg')

        A = cp.abs( probe_update_denominator[ tt, ... ] )
        fig, ax1 = plt.subplots( nrows = 1, ncols = 1, )
        pos1 = ax1.imshow( A.get(), cmap = 'gray', ) 
        plt.colorbar(pos1)
        plt.show( block = False )
        
        '''

    tike.communicators.stream.stream_and_modify2(       # stream over scan positions
        f=make_certain_args_constant,
        ind_args=[],
        streams=streams,
        lo=0,
        hi=len(scan),
    )

    return probe_update_denominator


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
