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
    and the current calculated step, options are 0.0 <= step_length_weight <= 1.0."""

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

    unmeasured_pixels: typing.Union[npt.NDArray, None] = dataclasses.field(
        default_factory=lambda: np.zeros(1, dtype=float))
    """A binary array that defines spatial regions on the detector where we have unmeasured pixels."""

    measured_pixels: typing.Union[npt.NDArray, None] = dataclasses.field(
        default_factory=lambda: np.ones(1, dtype=float))
    """A binary array that defines spatial regions on the detector where we have measured pixels."""

    def copy_to_device(self, comm):
        """Copy to the current GPU memory."""
        if self.unmeasured_pixels is not None:
            self.unmeasured_pixels = cp.asarray(self.unmeasured_pixels)
        if self.measured_pixels is not None:
            self.measured_pixels = cp.asarray(self.measured_pixels)
        # if self.unmeasured_pixels is not None:
        #     self.unmeasured_pixels = comm.pool.bcast([self.unmeasured_pixels])
        return self

    def copy_to_host(self):
        """Copy to the host CPU memory."""
        if self.unmeasured_pixels is not None:
            self.unmeasured_pixels = cp.asnumpy(self.unmeasured_pixels)
        if self.measured_pixels is not None:
            self.measured_pixels = cp.asnumpy(self.measured_pixels)
        # if self.unmeasured_pixels is not None:
        #     self.unmeasured_pixels = cp.asnumpy( self.unmeasured_pixels[0] )
        return self

    def resample(self, factor: float) -> ExitWaveOptions:
        """Return a new `ExitWaveOptions` with the parameters rescaled."""
        # FIXME: Unmeasured/measured pixels should be cropped/padded during
        # resampling
        return ExitWaveOptions(
            noise_model=self.noise_model,
            step_length_weight=self.step_length_weight,
            step_length_start=self.step_length_start,
            step_length_usemodes=self.step_length_usemodes,
            unmeasured_pixels=self.unmeasured_pixels,
            measured_pixels=self.measured_pixels,
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

        xi_alpha_minus_one = xi * step_length[..., None, None] - 1

        numer = I_m * xi_alpha_minus_one
        denom = abs2_Psi * cp.square(xi_alpha_minus_one) + I_e - abs2_Psi
        numer = cp.sum(measured_pixels * xi_abs_Psi2 * (1 + numer / denom),
                       axis=(-1, -2))

        denom = cp.sum(measured_pixels * cp.square(xi) * abs2_Psi,
                       axis=(-1, -2))

        step_length = step_length * (1 - weight_avg) + (numer /
                                                        denom) * weight_avg

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

    denom = measured_pixels * I_e * cp.square(xi)

    sum_denom = cp.sum(denom, axis=(-1, -2))

    for _ in range(0, 2):

        nom = measured_pixels * xi * (I_e - I_m /
                                      (1 - step_length[..., None, None] * xi))

        nom_over_denom = cp.sum(nom, axis=(-1, -2)) / sum_denom

        step_length = (1 -
                       weight_avg) * step_length + weight_avg * nom_over_denom

        step_length = cp.abs(cp.fmax(cp.fmin(step_length, 1), 0))

    step_length += 1e-2 * tike.random.randomizer_cp.standard_normal(
        size=step_length.shape,
        dtype=tike.precision.floating,
    )

    return step_length