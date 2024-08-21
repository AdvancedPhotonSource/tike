"""Defines a free-space propagation operator based on the CuPy FFT module."""

__author__ = "Ashish Tripathi"
__copyright__ = "Copyright (c) 2024, UChicago Argonne, LLC."

import typing

import numpy.typing as npt
import numpy as np

from .cache import CachedFFT
from .operator import Operator


class FresnelSpectProp(CachedFFT, Operator):
    """Fresnel spectrum propagation (short range) using CuPy.

    Take an (..., W, H) compelx array representing a wavefront and propagate.

    Attributes
    ----------
    pixel_size : float
        The realspace size of a pixel in meters
    distance : float
        The realspace propagation distance in meters
    wavelength : float
        The wavelength of the light in meters

    Parameters
    ----------
    nearplane: (..., W, H) complex64
        The wavefronts before propagation.
    farplane: (..., W, H) complex64
        The wavefronts after propagation.
    """

    def __init__(
        self,
        norm: str = "ortho",
        pixel_size: float = 1e-7,
        probe_FOV: typing.Tuple[float, float] = ( 1e-6, 1e-6 ),
        distance: float = 1e-6,
        wavelength: float = 1e-9,
        **kwargs,
    ):
        self.norm = norm
        self.pixel_size = pixel_size
        self.probe_FOV = probe_FOV
        self.distance = distance
        self.wavelength = wavelength

    def fwd(
        self,
        nearplane: npt.NDArray[np.csingle],
        overwrite: bool = False,
        **kwargs,
    ) -> npt.NDArray[np.csingle]:
        """forward (parallel to beam direction) Fresnel spectrum propagtion operator"""
        propagator = self._create_fresnel_spectrum_propagator(
            (nearplane.shape[-2], nearplane.shape[-1]),
            self.probe_FOV,
            self.distance,
            self.wavelength,
        )

        nearplane_fft2 = self._fft2(
            nearplane,
            norm=self.norm,
            axes=(-2, -1),
            overwrite_x=overwrite,
        )

        farplane = self._ifft2(
            nearplane_fft2 * propagator,
            norm=self.norm,
            axes=(-2, -1),
            overwrite_x=overwrite,
        )

        return farplane

    def adj(
        self,
        farplane: npt.NDArray[np.csingle],
        overwrite: bool = False,
        **kwargs,
    ) -> npt.NDArray[np.csingle]:
        """backward (anti-parallel to beam direction) Fresnel spectrum propagtion operator"""
        propagator = self._create_fresnel_spectrum_propagator(
            (farplane.shape[-2], farplane.shape[-1]),
            self.probe_FOV,
            self.distance,
            self.wavelength,
        )

        farplane_fft2 = self._fft2(
            farplane,
            norm=self.norm,
            axes=(-2, -1),
            overwrite_x=overwrite,
        )

        nearplane = self._ifft2(
            # FIXME: IS IT OK TO ALWAYS TAKE CONJ? OR SHOULD WE DO THIS ONCE AND REUSE?
            farplane_fft2 * self.xp.conj(propagator),
            norm=self.norm,
            axes=(-2, -1),
            overwrite_x=overwrite,
        )

        return nearplane

    def _create_fresnel_spectrum_propagator(
        self,
        N: typing.Tuple[int, int],
        probe_FOV: float = 1.0,
        distance: float = 1.0,
        wavelength: float = 1.0,
    ) -> np.ndarray:
        
        # FIXME: Check that dimension ordering is consistent

        xgrid = ( 0.5 + self.xp.linspace( ( -0.5 * N[1] ), ( 0.5 * N[1] - 1 ), num = N[1] )) / N[1]
        ygrid = ( 0.5 + self.xp.linspace( ( -0.5 * N[0] ), ( 0.5 * N[0] - 1 ), num = N[0] )) / N[0]

        kx = 2 * self.xp.pi * N[1] * xgrid / probe_FOV[ 1 ]
        ky = 2 * self.xp.pi * N[0] * ygrid / probe_FOV[ 0 ]

        Kx, Ky = self.xp.meshgrid(kx, ky, indexing='xy')

        fresnel_spectrum_propagator = self.xp.exp( 1j * distance * self.xp.sqrt( ( 2 * self.xp.pi / wavelength ) ** 2 - Kx ** 2 - Ky ** 2 ))

        fresnel_spectrum_propagator = self.xp.ndarray.astype( self.xp.fft.fftshift( fresnel_spectrum_propagator ), dtype = self.xp.csingle )

        return fresnel_spectrum_propagator


'''
def create_fresnel_spectrum_propagator( 
        N: np.ndarray,                                      # probe dimensions ( WIDE, HIGH )
        beam_energy: float,                                 # x-ray energy ( eV )
        delta_z: float,                                     # meters
        detector_dist: float,                               # meters
        detector_pixel_width: float ) -> np.ndarray:        # meters

    wavelength = ( 1.23984193e-9 / ( beam_energy / 1e3 ))       # x-ray energy ( eV ), wavelength ( meters )

    xgrid = ( 0.5 + np.linspace( ( -0.5 * N[1] ), ( 0.5 * N[1] - 1 ), num = N[1] )) / N[1]
    ygrid = ( 0.5 + np.linspace( ( -0.5 * N[0] ), ( 0.5 * N[0] - 1 ), num = N[0] )) / N[0]

    x       = wavelength * detector_dist / detector_pixel_width 
    #z_obj_L = np.asarray( [ x, x ], dtype = np.float32 )
    z_obj_L = np.asarray( [ x, x ], dtype = np.float64 )

    kx = 2 * np.pi * N[0] * xgrid / z_obj_L[ 0 ]
    ky = 2 * np.pi * N[1] * ygrid / z_obj_L[ 1 ]

    Kx, Ky = np.meshgrid(kx, ky, indexing='xy')

    fresnel_spectrum_propagator = np.exp( 1j * delta_z * np.sqrt( ( 2 * np.pi / wavelength ) ** 2 - Kx ** 2 - Ky ** 2 ))

    fresnel_spectrum_propagator = np.ndarray.astype( np.fft.fftshift( fresnel_spectrum_propagator ), dtype = np.csingle )

    return fresnel_spectrum_propagator

'''


    # def _create_fresnel_spectrum_propagator(
    #     self,
    #     N: typing.Tuple[int, int],
    #     pixel_size: float = 1.0,
    #     distance: float = 1.0,
    #     wavelength: float = 1.0,
    # ) -> np.ndarray:
    #     # FIXME: Check that dimension ordering is consistent
    #     rr2 = self.xp.linspace(-0.5 * N[1], 0.5 * N[1] - 1, num=N[1]) ** 2
    #     cc2 = self.xp.linspace(-0.5 * N[0], 0.5 * N[0] - 1, num=N[0]) ** 2

    #     x = -1j * self.xp.pi * wavelength * distance
    #     rr2 = self.xp.exp(x * rr2[..., None] / (pixel_size**2))
    #     cc2 = self.xp.exp(x * cc2[..., None] / (pixel_size**2))

    #     fresnel_spectrum_propagator = self.xp.ndarray.astype(
    #         self.xp.fft.fftshift(self.xp.outer(self.xp.transpose(rr2), cc2)),
    #         dtype=self.xp.csingle,
    #     )

    #     return fresnel_spectrum_propagator



