"""Functions related to creating and manipulating ptychographic exitwave arrays.

Ptychographic exitwaves are stored as a single complex array which represent
the wavefield after any and all interaction with the sample and thus there's
just free space propagation to the detector.

"""

from __future__ import annotations
import dataclasses
import logging
import typing

import cupy as cp
import numpy as np
import numpy.typing as npt

import tike.random

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ExitWaveOptions:
    """Manage data and setting related to exitwave updates."""

    noise_model: str = "gaussian" 
    """This string sets which noise model we use for the exitwave updates, 
    options are = { "gaussian", "poisson" }."""

    step_length_weight: float = 0.5 
    """When computing steplength, we use a weighted average of the previous calculated step 
    and the current calculated step, options are 0.0 <= step_length_weight <= 1.0, with being
    closer to 1.0 favoring the current calculated step"""

    step_length_usemodes: str = "all_modes" 
    """When computing steplength for exitwave updates, there are two ways we do this:
    1) use the dominant mode to compute the steplength and use that steplength for the other
    less dominant modes, and 2) compute the steplength for each mode independently, 
    options are { "dominant_mode", "all_modes" }."""

    step_length_start: float = 0.5
    """We use an iterative/recursive method for finding the steplengths, 
    and this is what we use as initialization for that method."""

    unmeasured_pixels_scaling: float = 1.00
    """Depending on how we control scaling for the exitwaves, we might need to scale up or down 
    the number of photons in the unmeasured regions for the exitwave updates in Fourier space."""

    measured_pixels: typing.Union[npt.NDArray, None] = dataclasses.field(
        default_factory=lambda: np.ones(1, dtype=bool))
    """A binary array that defines spatial regions on the detector where we have measured pixels."""

    def copy_to_device(self, comm):
        """Copy to the current GPU memory."""
        if self.measured_pixels is not None:
            self.measured_pixels = cp.asarray(self.measured_pixels)
        # if self.measured_pixels is not None:
        #     self.measured_pixels = comm.pool.bcast([self.measured_pixels])
        return self

    def copy_to_host(self):
        """Copy to the host CPU memory."""
        if self.measured_pixels is not None:
            self.measured_pixels = cp.asnumpy(self.measured_pixels)
        # if self.measured_pixels is not None:
        #     self.measured_pixels = cp.asnumpy( self.measured_pixels[0] )
        return self

    def resample(self, factor: float) -> ExitWaveOptions:
        """Return a new `ExitWaveOptions` with the parameters rescaled."""
        return ExitWaveOptions(
            noise_model=self.noise_model,
            step_length_weight=self.step_length_weight,
            step_length_start=self.step_length_start,
            step_length_usemodes=self.step_length_usemodes,
            measured_pixels=crop_fourier_space(
                self.measured_pixels,
                int(self.measured_pixels.shape[-1] * factor),
            ),
            unmeasured_pixels_scaling=self.unmeasured_pixels_scaling)


def poisson_steplength_all_modes(
    xi,
    abs2_Psi,
    I_e,
    I_m,
    measured_pixels,
    step_length,
    weight_avg,
):
    """Compute the optimal steplength for each exitwave mode independently.

    Parameters
    ----------
    xi              : ND array ( # scan positions, # rows, # columns ), xi = 1 - I_m / I_e
    abs2_Psi        : ND array ( # shared probe modes, # scan positions, # rows, # columns ), the squared absolute value of the calulated exitwaves       
    I_m             : ND array ( # scan positions, # rows, # columns ), measured diffraction intensity
    I_e             : ND array ( # scan positions, # rows, # columns ), calculated diffraction intensity 
    measured_pixels : 2D binary array ( # rows, # columns ), the regions on the detector where we have defined measurements
    step_length     : 2D array ( # shared probe modes, # scan positions), the steplength initializations
    weight_avg      : float ( 0.0 <= weight_avg <= 1.0  ), the weight we use when computing a weighted average
    """

    if measured_pixels.size == 0:
        measured_pixels = 1

    xi_abs_Psi2 = xi * abs2_Psi

    for _ in range(0, 2):

        xi_alpha_minus_one =  ( xi * step_length[..., None, None] - 1 )[ :, None, ... ]

        numer = I_m * xi_alpha_minus_one
        denom = abs2_Psi * cp.square(xi_alpha_minus_one) + I_e - abs2_Psi
        numer = cp.sum(measured_pixels * xi_abs_Psi2 * (1 + numer / denom),
                       axis=(-1, -2))

        denom = cp.sum(measured_pixels * cp.square(xi) * abs2_Psi,
                       axis=(-1, -2))

        step_length = step_length * (1 - weight_avg) + (numer[:, 0, :] /
                                                        denom[:, 0, :]) * weight_avg

    return step_length


def poisson_steplength_dominant_mode(
    xi,
    I_e,
    I_m,
    measured_pixels,
    step_length,
    weight_avg,
):
    """Compute the optimal steplength for each exitwave mode using only the dominant mode.

    Parameters
    ----------
    xi              : ND array ( # scan positions, # rows, # columns ), xi = 1 - I_m / I_e
    I_m             : ND array ( # scan positions, # rows, # columns ), measured diffraction intensity
    I_e             : ND array ( # scan positions, # rows, # columns ), calculated diffraction intensity 
    measured_pixels : 2D binary array ( # rows, # columns ), the regions on the detector where we have defined measurements
    step_length     : 2D array ( # shared probe modes, # scan positions), the steplength initializations
    weight_avg      : float ( 0.0 <= weight_avg <= 1.0  ), the weight we use when computing a weighted average
    """

    if measured_pixels.size == 0:
        measured_pixels = 1

    sum_denom = cp.sum(measured_pixels * I_e * cp.square(xi), axis=(-1, -2))

    for _ in range(0, 2):

        nom = measured_pixels * xi * (I_e - I_m /
                                      (1 - step_length[..., None, None] * xi))

        nom_over_denom = cp.sum(nom, axis=(-1, -2)) / sum_denom

        step_length = (1 -
                       weight_avg) * step_length + weight_avg * nom_over_denom

        # step_length = cp.abs(cp.fmax(cp.fmin(step_length, 1), 0))

    return step_length


def crop_fourier_space(x: np.ndarray, w: int) -> np.ndarray:
    """Crop x assuming a 2D frequency space image with zero frequency in corner."""
    assert x.shape[-2] == x.shape[-1], "Only works on square arrays right now."
    half1 = w // 2
    half0 = w - half1
    # yapf: disable
    return x[
        ..., np.r_[0:half0, (x.shape[-1] - half1):x.shape[-1]],
    ][
        ..., np.r_[0:half0, (x.shape[-2] - half1):x.shape[-2]], :,
    ]
    # yapf: enable
