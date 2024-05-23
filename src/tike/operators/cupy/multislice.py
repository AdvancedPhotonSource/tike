"""Defines a free-space propagation operator based on the CuPy FFT module."""

__author__ = "Ashish Tripathi"
__copyright__ = "Copyright (c) 2024, UChicago Argonne, LLC."

import typing
import numpy.typing as npt
import numpy as np
import cupy as cp

import tike.operators.cupy.objective as objective

from .cache import CachedFFT
from .operator import Operator

from .fresnelspectprop import FresnelSpectProp
from .convolution import Convolution  

class Multislice(CachedFFT, Operator):
    """Multislice wavefield propagation starting from a 2D probe through a 3D object using CuPy.

    So far, it's based on Fresnel spectrum propagation (short range) diffraction based. 

    !!!!! Take an (..., N, N) array and apply the Fourier transform to the last two
    dimensions. !!!!!

    Attributes
    ----------
    detector_shape : int
        The pixel width and height of the nearplane and farplane waves.

    multislice_propagator : (..., detector_shape, detector_shape) complex64
        The wavefield propagator 2D matrix that propagates one slice to the next,
        for backwards wavefield propagation we just take complex conjugate.

    Parameters
    ----------
    nearplane: (..., detector_shape, detector_shape) complex64
        The wavefronts after exiting the object.
    farplane: (..., detector_shape, detector_shape) complex64
        The wavefronts hitting the detector respectively. Shape for cost
        functions and gradients is (nscan, 1, 1, detector_shape,
        detector_shape).
    multislice_inputplane: (..., detector_shape, detector_shape) complex64
        The intermediate (between the probe plane and nearplane) wavefronts 
        that are numerically propagated within the 3D sample

    .. versionchanged:: 0.25.0 Removed the model parameter and the cost(),
        grad() functions. Use the cost and gradient functions directly instead.

    """

    def __init__(
        self,
        detector_shape: int,
        probe_shape: int,
        nz: int,
        n: int,
        multislice_total_slices: int,
        multislice_propagator: npt.NDArray[np.csingle],
        diffraction: typing.Type[Convolution] = Convolution,
        fresnelspectprop: typing.Type[FresnelSpectProp] = FresnelSpectProp,
        norm: str = 'ortho',
        **kwargs,
    ):

        self.diffraction = diffraction(                 # extract 2D slices from object and form exitwaves
            probe_shape=probe_shape,
            detector_shape=detector_shape,
            nz=nz,
            n=n,
            **kwargs,
        )
        self.fresnelspectprop = fresnelspectprop(                   # propagate through 3D sample using 2D probe 
            detector_shape=detector_shape,
            norm=norm,
            multislice_propagator = multislice_propagator,
            **kwargs,
        )

        # TODO: Replace these with @property functions
        self.probe_shape = probe_shape
        self.detector_shape = detector_shape
        self.nz = nz
        self.n = n
        self.multislice_total_slices = multislice_total_slices

    def __enter__(self):
        self.diffraction.__enter__()    
        self.fresnelspectprop.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self.diffraction.__exit__(type, value, traceback)
        self.fresnelspectprop.__exit__(type, value, traceback)

    def fwd(
        self,
        psi, 
        scan, 
        probe, 
        overwrite: bool = False,
        **kwargs,
    ) -> npt.NDArray[np.csingle]:           # forward (parallel to beam direction) multislice problem

        multislice_probes = cp.zeros( ( psi.shape[0], scan.shape[-2], *probe.shape[-3:] ), dtype = cp.csingle )
        multislice_probes[ 0, ... ] = probe[..., 0, :, :, :]            # = cp.repeat( probe, scan.shape[0], axis = 0)[..., 0, :, :, :]
 
        for tt in cp.arange( 0, psi.shape[0], 1 ) :

            multislice_exwv = self.diffraction.fwd(
                    psi   = psi[ tt, ... ],               
                    scan  = scan,
                    probe = multislice_probes[ tt, ... ],
                )
            
            if tt == ( psi.shape[0] - 1 ) :
                break

            multislice_probes[ tt + 1, ... ] = self.fresnelspectprop.fwd(         
                    multislice_inputplane = multislice_exwv,
                    multislice_propagator = self.fresnelspectprop.multislice_propagator, 
                    overwrite=False,                
                )

        return multislice_exwv, multislice_probes

    def adj(
        self,
        psi, 
        scan, 
        probe, 
        diff,
        multislice_probes,
        overwrite: bool = False,
        **kwargs,
    ) -> npt.NDArray[np.csingle]:           # backward (anti-parallel to beam direction) multislice propagation of exitwave update
          
        # self._check_shape(multislice_outputplane)
        # shape = multislice_outputplane.shape

        # multislice_exwv_update = cp.zeros( ( psi.shape[0], scan.shape[-2], *probe.shape[-3:] ), dtype = cp.csingle )
        # multislice_exwv_update[ 0, ... ] = diff[..., 0, :, :, :]            # = cp.repeat( probe, scan.shape[0], axis = 0)[..., 0, :, :, :]
 
        # for tt in cp.arange( 0, psi.shape[0], 1 ) :

        #     multislice_exwv_update_fft2 = self._fft2( 
        #         multislice_exwv_update[ tt, ... ],
        #         norm=self.norm,
        #         axes=(-2, -1),
        #         overwrite_x=overwrite,
        #     )

        #     multislice_inputplane = self._ifft2(
        #         multislice_exwv_update_fft2 * cp.conj( self.fresnelspectprop.multislice_propagator ),        # IS IT OK TO ALWAYS TAKE CONJ? OR SHOULD WE DO THIS ONCE AND REUSE?
        #         norm=self.norm,
        #         axes=(-2, -1),
        #         overwrite_x=overwrite,
        #     )

        multislice_exwv_update = 0
        return multislice_exwv_update

    def _check_shape(self, x: npt.NDArray) -> None:
        assert type(x) is self.xp.ndarray, type(x)
        shape = (-1, self.detector_shape, self.detector_shape)
        if __debug__ and x.shape[-2:] != shape[-2:]:
            raise ValueError(f"waves must have shape {shape} not {x.shape}.")