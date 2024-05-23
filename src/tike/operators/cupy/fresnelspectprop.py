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

    Take an (..., N, N) array and apply the Fourier transform to the last two
    dimensions.

    Attributes
    ----------
    pixel_size : float
        The realspace size of a pixel in meters
    delta_z : float
        The realspace propagation distance in meters
    wavelength : float
        The wavelength of the light in meters

    Parameters
    ----------
    nearplane: (..., detector_shape, detector_shape) complex64
        The wavefronts before propagation.
    farplane: (..., detector_shape, detector_shape) complex64
        The wavefronts after propagation.
    """

    def __init__(
        self,
        norm: str = "ortho",
        pixel_size: float = 1.0,
        delta_z: float = 1.0,
        wavelength: float = 1.0,
        **kwargs,
    ):
        self.norm = norm
        self.pixel_size = pixel_size
        self.delta_z = delta_z
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
            self.pixel_size,
            self.delta_z,
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
            self.pixel_size,
            self.delta_z,
            self.wavelength,
        )

        farplane_fft2 = self._fft2(
            farplane,
            norm=self.norm,
            axes=(-2, -1),
            overwrite_x=overwrite,
        )

        nearplane = self._ifft2(
            farplane_fft2
            * self.xp.conj(
                propagator,
            ),  # IS IT OK TO ALWAYS TAKE CONJ? OR SHOULD WE DO THIS ONCE AND REUSE?
            norm=self.norm,
            axes=(-2, -1),
            overwrite_x=overwrite,
        )

        return nearplane

    def _create_fresnel_spectrum_propagator(
        self,
        N: typing.Tuple[int, int],
        pixel_size: float = 1.0,
        delta_z: float = 1.0,
        wavelength: float = 1.0,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        pixel_size : real width of pixel in meters
        delta_z: propagation distance in meters
        wavelength: wavelength of light in meters
        """
        # FIXME: Check that dimension ordering is consistent
        rr2 = self.xp.linspace(-0.5 * N[1], 0.5 * N[1] - 1, num=N[1]) ** 2
        cc2 = self.xp.linspace(-0.5 * N[0], 0.5 * N[0] - 1, num=N[0]) ** 2

        prb_FOV = self.xp.asarray([pixel_size, pixel_size], dtype=self.xp.float32)

        x = -1j * self.xp.pi * wavelength * delta_z
        rr2 = self.xp.exp(x * rr2[..., None] / (prb_FOV[0] ** 2))
        cc2 = self.xp.exp(x * cc2[..., None] / (prb_FOV[1] ** 2))

        fresnel_spectrum_propagator = self.xp.ndarray.astype(
            self.xp.fft.fftshift(self.xp.outer(self.xp.transpose(rr2), cc2)),
            dtype=self.xp.csingle,
        )

        return fresnel_spectrum_propagator
