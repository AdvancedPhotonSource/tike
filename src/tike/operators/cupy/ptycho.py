"""Defines a ptychography operator based on the CuPy FFT module."""

__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

import typing

import numpy.typing as npt
import numpy as np
import cupy as cp

from .operator import Operator
from .propagation import Propagation
from .convolution import Convolution
from .multislice_fresnelspectprop import Multislice_FresnelSpectProp as Multislice

from . import objective


@cp.fuse()
def _intensity_from_farplane(farplane):
    return cp.sum(
        cp.real(farplane * cp.conj(farplane)),
        axis=tuple(range(1, farplane.ndim - 2)),
    )


class Ptycho(Operator):
    """A Ptychography operator.

    Compose a diffraction and propagation operator to simulate the interaction
    of an illumination wavefront with an object followed by the propagation of
    the wavefront to a detector plane.


    Parameters
    ----------
    detector_shape : int
        The pixel width and height of the (square) detector grid.
    nz, n : int
        The pixel width and height of the 2D reconstructed grid.
    multislice_total_slices: uint
       The number of slices we use for multislice ptycho.
    probe_shape : int
        The pixel width and height of the (square) probe illumination.
    propagation : :py:class:`Operator`
        The farfield free space wave propagation operator.
    diffraction : :py:class:`Operator`
        The projection approximation (2D exitwave = 2D object * 2D probe) operator.
    multislice : :py:class:`Operator`
        The multislice operator which computes 2D exitwaves by propagating a 2D probe through a 3D object.
    data : (..., FRAME, WIDE, HIGH) float32
        The intensity (square of the absolute value) of the propagated
        wavefront; i.e. what the detector records.
    farplane: (..., POSI, 1, SHARED, detector_shape, detector_shape) complex64
        The wavefronts hitting the detector respectively.
    probe : {(..., 1, 1, SHARED, WIDE, HIGH), (..., POSI, 1, SHARED, WIDE, HIGH)} complex64
        The complex illumination function.
    psi : (..., WIDE, HIGH) complex64
        The wavefront modulation coefficients of the object.
    scan : (..., POSI, 2) float32
        Coordinates of the minimum corner of the probe grid for each
        measurement in the coordinate system of psi. Coordinate order
        consistent with WIDE, HIGH order.
    multislice_propagator : (..., WIDE, HIGH) complex64
        A 2D array that numerically propagates a 2D wavefield from an input plane to an output plane

    .. versionchanged:: 0.25.0 Removed the model and ntheta parameters.

    """

    def __init__(
        self,
        detector_shape: int,
        probe_shape: int,
        nz: int,
        n: int,
        multislice_total_slices: int,
        multislice_propagator: npt.NDArray[np.csingle],
        propagation: typing.Type[Propagation] = Propagation,
        diffraction: typing.Type[Convolution] = Convolution,
        multislice: typing.Type[Multislice] = Multislice,
        norm: str = 'ortho',
        **kwargs,
    ):
        """Please see help(Ptycho) for more info."""
        self.propagation = propagation(             
            detector_shape=detector_shape,
            norm=norm,
            **kwargs,
        )
        self.diffraction = diffraction(
            probe_shape=probe_shape,
            detector_shape=detector_shape,
            nz=nz,
            n=n,
            **kwargs,
        )
        self.multislice = multislice(
            norm=norm,
            multislice_propagator = multislice_propagator,
        )
        # TODO: Replace these with @property functions
        self.probe_shape = probe_shape
        self.detector_shape = detector_shape
        self.nz = nz
        self.n = n
        self.multislice_total_slices = multislice_total_slices

    def __enter__(self):
        self.propagation.__enter__()
        self.diffraction.__enter__()
        self.multislice.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self.propagation.__exit__(type, value, traceback)
        self.diffraction.__exit__(type, value, traceback)
        self.multislice.__exit__(type, value, traceback)

    def fwd(
        self,
        probe: npt.NDArray[np.csingle],
        scan: npt.NDArray[np.single],
        psi: npt.NDArray[np.csingle],
        **kwargs,
    ) -> npt.NDArray[np.csingle]:
        """Please see help(Ptycho) for more info."""

        # BEFORE:
        #
        # return self.propagation.fwd(
        #     self.diffraction.fwd(
        #         psi=psi,                          #  cp.mean( psi, axis = 0 )
        #         scan=scan,    
        #         probe=probe[..., 0, :, :, :],
        #     ),
        #     overwrite=True,
        # )[..., None, :, :, :]
    
  

        multislice_probes = cp.zeros( ( psi.shape[0], scan.shape[-2], *probe.shape[-3:] ), dtype = cp.csingle )
        multislice_probes[ 0, ... ] = probe[..., 0, :, :, :]            # = cp.repeat( probe, scan.shape[0], axis = 0)[..., 0, :, :, :]
 
        for tt in cp.arange( 0, psi.shape[0], 1 ) :

            multislice_exwvs = self.diffraction.fwd(
                    psi   = psi[ tt, ... ],               
                    scan  = scan,
                    probe = multislice_probes[ tt, ... ],
                )
            
            if tt == ( psi.shape[0] - 1 ) :
                break

            multislice_probes[ tt + 1, ... ] = self.multislice.fwd(
                    multislice_inputplane = multislice_exwvs,
                    multislice_propagator = self.multislice.multislice_propagator, 
                    overwrite=False,                
                )

        multislice_farfield = self.propagation.fwd(
            multislice_exwvs,
            overwrite=True,                     
        )
        multislice_farfield = multislice_farfield[..., None, :, :, :]


        ''' 

        import matplotlib.pyplot as plt
        #import numpy as np
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib as mpl
        # mpl.use('Agg')
        mpl.use('TKAgg')

        pp = 0
        ss = 13

        A = np.abs( multislice_probes[ 0, ss, pp, ... ] )
        fig, ax1 = plt.subplots( nrows = 1, ncols = 1, )
        pos1 = ax1.imshow( A.get(), cmap = 'gray', ) 
        plt.colorbar(pos1)
        plt.show( block = False )
    
        B = np.abs( multislice_probes[ 1, ss, pp, ... ] )
        fig, ax2 = plt.subplots( nrows = 1, ncols = 1, )
        pos2 = ax2.imshow( B.get(), cmap = 'gray', ) 
        plt.colorbar(pos2)
        plt.show( block = False )

        C = np.abs( multislice_probes[ 2, ss, pp, ... ] )
        fig, ax3 = plt.subplots( nrows = 1, ncols = 1, )
        pos3 = ax3.imshow( C.get(), cmap = 'gray', ) 
        plt.colorbar(pos3)
        plt.show( block = False )

        '''

        return multislice_farfield, multislice_probes
    
    def adj(
        self,
        farplane: npt.NDArray[np.csingle],
        probe: npt.NDArray[np.csingle],
        scan: npt.NDArray[np.single],
        psi: npt.NDArray[np.csingle] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> npt.NDArray[np.csingle]:
        """Please see help(Ptycho) for more info."""
        return self.diffraction.adj(
            nearplane=self.propagation.adj(     # propagate farplane  to 
                farplane,
                overwrite=overwrite,
            )[..., 0, :, :, :],
            probe=probe[..., 0, :, :, :],
            scan=scan,
            overwrite=True,
            psi=psi,
        )

    def adj_probe(
        self,
        farplane: npt.NDArray[np.csingle],
        scan: npt.NDArray[np.single],
        psi: npt.NDArray[np.csingle],
        overwrite: bool = False,
        **kwargs,
    ) -> npt.NDArray[np.csingle]:
        """Please see help(Ptycho) for more info."""
        return self.diffraction.adj_probe(
            psi=psi,
            scan=scan,
            nearplane=self.propagation.adj(
                farplane=farplane,
                overwrite=overwrite,
            )[..., 0, :, :, :],
            overwrite=True,
        )[..., None, :, :, :]

    def _compute_intensity(
        self,
        data: npt.NDArray,
        psi: npt.NDArray[np.csingle],
        scan: npt.NDArray[np.single],
        probe: npt.NDArray[np.csingle],
    ) -> npt.NDArray[np.single]:
        """Compute detector intensities replacing the nth probe mode"""
        farplane, _ = self.fwd(         
            psi=psi,
            scan=scan,
            probe=probe,
        )
        return _intensity_from_farplane(farplane), farplane
    
    def cost(
        self,
        data: npt.NDArray,
        psi: npt.NDArray[np.csingle],
        scan: npt.NDArray[np.single],
        probe: npt.NDArray[np.csingle],
        *,
        model: str,
    ) -> float:
        """Please see help(Ptycho) for more info."""
        intensity, _ = self._compute_intensity(data, psi, scan, probe)
        return getattr(objective, model)(data, intensity)

    def grad_psi(
        self,
        data: npt.NDArray,
        psi: npt.NDArray[np.csingle],
        scan: npt.NDArray[np.single],
        probe: npt.NDArray[np.csingle],
        *,
        model: str,
    ) -> npt.NDArray[np.csingle]:
        """Please see help(Ptycho) for more info."""
        intensity, farplane = self._compute_intensity(data, psi, scan, probe)
        grad_obj = self.xp.zeros_like(psi)
        grad_obj = self.adj(
            farplane=getattr(objective, f'{model}_grad')(
                data,
                farplane,
                intensity,
            ),
            probe=probe,
            scan=scan,
            psi=grad_obj,
            overwrite=True,
        )
        return grad_obj

    def grad_probe(
        self,
        data: npt.NDArray,
        psi: npt.NDArray[np.csingle],
        scan: npt.NDArray[np.single],
        probe: npt.NDArray[np.csingle],
        mode: typing.List[int] = None,
        *,
        model: str,
    ) -> npt.NDArray[np.csingle]:
        """Compute the gradient with respect to the probe(s).

        Parameters
        ----------
        mode : list(int)
            Only return the gradient with resepect to these probes.

        """
        mode = list(range(probe.shape[-3])) if mode is None else mode
        intensity, farplane = self._compute_intensity(data, psi, scan, probe)
        # Use the average gradient for all probe positions
        return self.xp.mean(
            self.adj_probe(
                farplane=getattr(objective, f'{model}_grad')(
                    data,
                    farplane[..., mode, :, :],
                    intensity,
                ),
                psi=psi,
                scan=scan,
                overwrite=True,
            ),
            axis=0,
            keepdims=True,
        )

    def adj_all(
        self,
        farplane: npt.NDArray[np.csingle],
        probe: npt.NDArray[np.csingle],
        scan: npt.NDArray[np.single],
        psi: npt.NDArray[np.csingle],
        overwrite: bool = False,
        rpie: bool = False,
    ) -> typing.Tuple[npt.NDArray, ...]:
        """Please see help(Ptycho) for more info."""
        result = self.diffraction.adj_all(
            nearplane=self.propagation.adj(
                farplane,
                overwrite=overwrite,
            )[..., 0, :, :, :],
            probe=probe[..., 0, :, :, :],
            scan=scan,
            overwrite=True,
            psi=psi,
            rpie=rpie,
        )
        return (result[0], result[1][..., None, :, :, :], *result[2:])




'''

    def __init__(
        self,
        detector_shape: int,
        probe_shape: int,
        nz: int,
        n: int,
        multislice_total_slices: int,
        multislice_propagator: npt.NDArray[np.csingle],
        propagation: typing.Type[Propagation] = Propagation,
        diffraction: typing.Type[Convolution] = Convolution,
        norm: str = 'ortho',
        **kwargs,
    ):
        """Please see help(Ptycho) for more info."""
        self.propagation = propagation(             
            detector_shape=detector_shape,
            multislice_propagator = multislice_propagator,
            norm=norm,
            **kwargs,
        )
        self.diffraction = diffraction(
            probe_shape=probe_shape,
            detector_shape=detector_shape,
            nz=nz,
            n=n,
            **kwargs,
        )
        # TODO: Replace these with @property functions
        self.probe_shape = probe_shape
        self.detector_shape = detector_shape
        self.nz = nz
        self.n = n
        self.multislice_total_slices = multislice_total_slices

'''