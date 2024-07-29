"""Functions related to creating and manipulating ptychographic object arrays.

Ptychographic objects are stored as a single complex array which represent
the complex refractive indices in the field of view.

"""
from __future__ import annotations
import dataclasses
import logging
import typing
import copy

import cupy as cp
import cupyx.scipy.ndimage
import numpy as np
import numpy.typing as npt
import scipy.interpolate

import tike.linalg
import tike.precision

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ObjectOptions:
    """Manage data and setting related to object correction."""

    convergence_tolerance: float = 0
    """Terminate reconstruction early when the mnorm of the object update is
    less than this value."""

    update_mnorm: typing.List[float] = dataclasses.field(
        init=False,
        default_factory=list,
    )
    """A record of the previous mnorms of the object update."""

    positivity_constraint: float = 0
    """This value is passed to the tike.ptycho.object.positivity_constraint
    function."""

    smoothness_constraint: float = 0
    """This value is passed to the tike.ptycho.object.smoothness_constraint
    function."""

    use_adaptive_moment: bool = False
    """Whether or not to use adaptive moment."""

    vdecay: float = 0.999
    """The proportion of the second moment that is previous second moments."""

    mdecay: float = 0.9
    """The proportion of the first moment that is previous first moments."""

    v: typing.Union[npt.NDArray, None] = dataclasses.field(
        init=False,
        default_factory=lambda: None,
    )
    """The second moment for adaptive moment."""

    m: typing.Union[npt.NDArray, None] = dataclasses.field(
        init=False,
        default_factory=lambda: None,
    )
    """The first moment for adaptive moment."""

    preconditioner: typing.Union[npt.NDArray, None] = dataclasses.field(
        init=False,
        default_factory=lambda: None,
    )
    """The magnitude of the illumination used for conditioning the object updates."""

    clip_magnitude: bool = False
    """Whether to force the object magnitude to remain <= 1."""

    def copy_to_device(self) -> ObjectOptions:
        """Copy to the current GPU memory."""
        options = ObjectOptions(
            convergence_tolerance=self.convergence_tolerance,
            positivity_constraint=self.positivity_constraint,
            smoothness_constraint=self.smoothness_constraint,
            use_adaptive_moment=self.use_adaptive_moment,
            vdecay=self.vdecay,
            mdecay=self.mdecay,
            clip_magnitude=self.clip_magnitude,
        )
        options.update_mnorm = copy.copy(self.update_mnorm)
        if self.v is not None:
            options.v = cp.asarray(
                self.v,
                dtype=tike.precision.cfloating
                if np.iscomplexobj(self.v)
                else tike.precision.floating,
            )
        if self.m is not None:
            options.m = cp.asarray(
                self.m,
                dtype=tike.precision.cfloating
                if np.iscomplexobj(self.m)
                else tike.precision.floating,
            )
        if self.preconditioner is not None:
            options.preconditioner = cp.asarray(
                self.preconditioner,
                dtype=tike.precision.cfloating,
            )
        return options

    def copy_to_host(self) -> ObjectOptions:
        """Copy to the host CPU memory."""
        options = ObjectOptions(
            convergence_tolerance=self.convergence_tolerance,
            positivity_constraint=self.positivity_constraint,
            smoothness_constraint=self.smoothness_constraint,
            use_adaptive_moment=self.use_adaptive_moment,
            vdecay=self.vdecay,
            mdecay=self.mdecay,
            clip_magnitude=self.clip_magnitude,
        )
        options.update_mnorm = copy.copy(self.update_mnorm)
        if self.v is not None:
            options.v = cp.asnumpy(self.v)
        if self.m is not None:
            options.m = cp.asnumpy(self.m)
        if self.preconditioner is not None:
            options.preconditioner = cp.asnumpy(self.preconditioner)
        return options

    def resample(self, factor: float, interp) -> ObjectOptions:
        """Return a new `ObjectOptions` with the parameters rescaled."""
        options = ObjectOptions(
            convergence_tolerance=self.convergence_tolerance,
            positivity_constraint=self.positivity_constraint,
            smoothness_constraint=self.smoothness_constraint,
            use_adaptive_moment=self.use_adaptive_moment,
            vdecay=self.vdecay,
            mdecay=self.mdecay,
            clip_magnitude=self.clip_magnitude,
        )
        options.update_mnorm = copy.copy(self.update_mnorm)
        return options
        # Momentum reset to zero when grid scale changes

    @staticmethod
    def join_psi(
        x: typing.List[np.ndarray],
        stripe_start: typing.List[int],
        probe_width: int,
    ) -> np.ndarray:
        """Recombine `x`, a list of psi, from a split reconstruction."""
        joined_psi = x[0]
        w = probe_width // 2
        for i in range(1, len(x)):
            lo = stripe_start[i] + w
            hi = stripe_start[i + 1] + w if i + 1 < len(x) else x[0].shape[1]
            joined_psi[:, lo:hi, :] = x[i][:, lo:hi, :]
        return joined_psi

    @staticmethod
    def join(
        x: typing.List[ObjectOptions],
        stripe_start: typing.List[int],
        probe_width: int,
    ) -> ObjectOptions:
        """Recombine `x`, a list of ObjectOptions, from split ObjectOptions"""
        options = ObjectOptions(
            convergence_tolerance=x[0].convergence_tolerance,
            positivity_constraint=x[0].positivity_constraint,
            smoothness_constraint=x[0].smoothness_constraint,
            use_adaptive_moment=x[0].use_adaptive_moment,
            vdecay=x[0].vdecay,
            mdecay=x[0].mdecay,
            clip_magnitude=x[0].clip_magnitude,
        )
        options.update_mnorm = copy.copy(x[0].update_mnorm)
        if x[0].v is not None:
            options.v = ObjectOptions.join_psi(
                [e.v for e in x],
                stripe_start,
                probe_width,
            )
        if x[0].m is not None:
            options.m = ObjectOptions.join_psi(
                [e.m for e in x],
                stripe_start,
                probe_width,
            )
        if x[0].preconditioner is not None:
            options.preconditioner = ObjectOptions.join_psi(
                [e.preconditioner for e in x],
                stripe_start,
                probe_width,
            )
        return options


def positivity_constraint(x, r):
    """Constrains the amplitude of x to be positive with sum of abs(x) and x.

    Parameters
    ----------
    r : float [0, 1]
        The proportion of abs(x) in the weighted sum of abs(x) and x.

    """
    if r > 0:
        if r > 1:
            raise ValueError(
                f"Positivity constraint must be in the range [0, 1] not {r}.")
        logger.info("Object positivity constrained with ratio %.3e", r)
        return r * cp.abs(x) + (1 - r) * x
    else:
        return x


def smoothness_constraint(x, a):
    """Convolves the image with a 3x3 averaging kernel.

    The kernel is defined as::

        [[a, a, a]
         [a, c, a]
         [a, a, a]]

    where c = 1 - 8 * a

    Parameters
    ----------
    a : float [0, 1/8)
        The non-center weights of the kernel.
    """
    if 0 <= a and a < 1.0 / 8.0:
        logger.info("Object smooth constrained with kernel param %.3e", a)
        weights = cp.ones([1] * (x.ndim - 2) + [3, 3],
                          dtype=tike.precision.floating) * a
        weights[..., 1, 1] = 1.0 - 8.0 * a
        x.real = cupyx.scipy.ndimage.convolve(x.real, weights, mode='nearest')
        x.imag = cupyx.scipy.ndimage.convolve(x.imag, weights, mode='nearest')
        return x
    else:
        raise ValueError(
            f"Smoothness constraint must be in range [0, 1/8) not {a}.")


def get_padded_object(scan, probe, extra: int = 0):
    """Return a ones-initialized object and shifted scan positions.

    An complex object array is initialized with shape such that the area
    covered by the probe is padded on each edge by a half probe width. The scan
    positions are shifted to be centered in this newly initialized object
    array.
    """
    int_scan = scan // 1
    min_corner = np.min(int_scan, axis=-2)
    max_corner = np.max(int_scan, axis=-2)
    span = max_corner - min_corner + probe.shape[-1] + 2 + 2 * extra
    return np.full_like(
        probe,
        shape=span.astype(tike.precision.integer),
        dtype=tike.precision.cfloating,
        fill_value=tike.precision.cfloating(0.5 + 0j),
    ), scan + 1 - min_corner + extra


def _int_min_max(x):
    """Return the integer range containing all points in the set x."""
    return np.floor(np.amin(x)), np.ceil(np.amax(x))


def get_absorbtion_image(data, scan, *, rescale=1.0, method='cubic'):
    """Approximate a scanning transmission image from diffraction patterns.

    Interpolates the diffraction patterns to a grid with spacing of one unit.

    Uses scipy.interpolate.griddata(), see docs for that function for more
    details about interpolation method.

    Parameters
    ----------
    data : (FRAME, WIDE, HIGH)
        The intensity (square of the absolute value) of the propagated
        wavefront i.e. what the detector records. FFT-shifted so the
        diffraction peak is at the corners.
    scan : (POSI, 2) float32
        Coordinates of the minimum corner of the probe grid for each
        measurement in the coordinate system of psi.
    rescale : float (0, 1.0]
        Rescale the scanning positions by this value before interpolating.
    method : str
        The interpolation method: linear, nearest, or cubic
    fill_value : float
        Value used to fill in for requested points outside of the convex hull
        of the input points. If not provided, then the default is nan. This
        option has no effect for the ‘nearest’ method.
    """
    rescaled = scan * rescale
    coord0, coord1 = np.meshgrid(
        np.arange(*_int_min_max(rescaled[:, 0])),
        np.arange(*_int_min_max(rescaled[:, 1])),
        indexing='ij',
    )
    values = np.square(tike.linalg.norm(data, axis=(-2, -1), keepdims=False))
    absorption_image = scipy.interpolate.griddata(
        points=rescaled,
        values=values,
        xi=(coord0.flatten(), coord1.flatten()),
        method=method,
        fill_value=np.amax(values),
    )
    return np.reshape(absorption_image, coord0.shape)


def remove_object_ambiguity(
    psi: npt.NDArray[cp.complex64],
    probe: npt.NDArray[cp.complex64],
    preconditioner: npt.NDArray[cp.complex64],
) -> typing.Tuple[npt.NDArray[cp.complex64], npt.NDArray[cp.complex64]]:
    """Remove normalization ambiguity between probe and psi"""
    W = preconditioner.real
    W = W / tike.linalg.mnorm(W)
    object_norm = 2 * np.sqrt(np.mean(np.square(np.abs(psi)) * W))
    psi = psi / object_norm
    probe = probe * object_norm
    return psi, probe
