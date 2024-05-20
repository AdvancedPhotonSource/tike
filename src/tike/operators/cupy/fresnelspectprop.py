"""Defines a free-space propagation operator based on the CuPy FFT module."""

__author__ = "Ashish Tripathi"
__copyright__ = "Copyright (c) 2024, UChicago Argonne, LLC."

import numpy.typing as npt
import numpy as np
import cupy as cp

import tike.operators.cupy.objective as objective

from .cache import CachedFFT
from .operator import Operator


class FresnelSpectProp(CachedFFT, Operator):
    """Fresnel spectrum propagation (short range) using CuPy.

    Take an (..., N, N) array and apply the Fourier transform to the last two
    dimensions.

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

    def __init__( self, norm: str = "ortho", multislice_propagator = None, **kwargs ):
        self.norm = norm
        self.multislice_propagator = multislice_propagator
        # self.multislice_propagator_conj = cp.conj( multislice_propagator )
 
    def fwd(
        self,
        multislice_inputplane: npt.NDArray[np.csingle],
        overwrite: bool = False,
        **kwargs,
    ) -> npt.NDArray[np.csingle]:           # forward (parallel to beam direction) Fresnel spectrum propagtion operator 

            # self._check_shape(multislice_inputplane)
            # shape = multislice_inputplane.shape

            multislice_inputplane_fft2 = self._fft2( 
                multislice_inputplane,
                norm=self.norm,
                axes=(-2, -1),
                overwrite_x=overwrite,
            )

            multislice_outputplane = self._ifft2(
                multislice_inputplane_fft2 * self.multislice_propagator,
                norm=self.norm,
                axes=(-2, -1),
                overwrite_x=overwrite,
            )

            return multislice_outputplane

    def adj(
        self,
        multislice_outputplane: npt.NDArray[np.csingle],
        overwrite: bool = False,
        **kwargs,
    ) -> npt.NDArray[np.csingle]:           # backward (anti-parallel to beam direction) Fresnel spectrum propagtion operator 
          
        # self._check_shape(multislice_outputplane)
        # shape = multislice_outputplane.shape

        multislice_outputplane_fft2 = self._fft2( 
            multislice_outputplane,
            norm=self.norm,
            axes=(-2, -1),
            overwrite_x=overwrite,
        )

        multislice_inputplane = self._ifft2(
            multislice_outputplane_fft2 * cp.conj( self.multislice_propagator ),        # IS IT OK TO ALWAYS TAKE CONJ? OR SHOULD WE DO THIS ONCE AND REUSE?
            norm=self.norm,
            axes=(-2, -1),
            overwrite_x=overwrite,
        )

        return multislice_inputplane

    # def _check_shape(self, x: npt.NDArray) -> None:
    #     assert type(x) is self.xp.ndarray, type(x)
    #     shape = (-1, self.detector_shape, self.detector_shape)
    #     if __debug__ and x.shape[-2:] != shape[-2:]:
    #         raise ValueError(f"waves must have shape {shape} not {x.shape}.")

def create_fresnel_spectrum_propagator( 
        N: np.ndarray,                                      # probe dimensions ( WIDE, HIGH )
        beam_energy: float,                                 # x-ray energy ( eV )
        delta_z: float,                                     # meters
        detector_dist: float,                               # meters
        detector_pixel_width: float ) -> np.ndarray:        # meters

    rr2 = np.linspace( -0.5 * N[1], 0.5 * N[1] - 1, num = N[1] ) ** 2
    cc2 = np.linspace( -0.5 * N[0], 0.5 * N[0] - 1, num = N[0] ) ** 2

    wavelength = ( 12.4 / ( beam_energy / 1e3 )) * 1e-10      # x-ray energy ( eV ), wavelength ( meters )

    x       = wavelength * detector_dist / detector_pixel_width 
    prb_FOV = np.asarray( [ x, x ], dtype = np.float32 )

    x = -1j * np.pi * wavelength * delta_z
    rr2 = np.exp( x * rr2[ ..., None ] / ( prb_FOV[0] ** 2 ))
    cc2 = np.exp( x * cc2[ ..., None ] / ( prb_FOV[1] ** 2 ))

    fresnel_spectrum_propagator = np.ndarray.astype( np.fft.fftshift( np.outer( np.transpose( rr2 ), cc2 )), dtype = np.csingle )

    return fresnel_spectrum_propagator