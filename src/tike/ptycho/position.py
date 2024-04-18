"""Functions for manipulating and updating scanning positions.

Tike coordinate system conventions explanation in one dimension
===============================================================

Minimum corner coordinates
--------------------------

Tike uses an origin-at-minimum-corner coordinate system. This means that
location of every object is described by its minimum corner (the corner closest
to negative infinity). It also means that valid coordinates (coordinates within
the field of view) are strictly non-negative because the field of view has its
origin at the minimum corner of the field of view, so none of the coordinates
within the field of view will be negative.

Tike uses pixel units for scan positions, so that there is a one-to-one-to-one
corespondence between scan positions, coordinates, and pixels in the field of
view.

::

    Figure 1: The coordinate system in the field of view

    [ 0 | 1 | 2 | 3 | 4 | .... ]
      ^           ^
      |           |
      -- This is the minimum corner of the field of view. Its coordinate is 0.
                  |
                  -- This is the 3rd pixel. Its coordinate is 3.

Because the coordinate system includes zero, a field of view with width `w`
will include coordinates 0, 1, 2, ..., (w-1). By extension, a probe located at
position 7 with a width 4 would cover coordinates 7, 8, 9, 10. The center of a
probe is located at its position plus half its width.

::

    Figure 2: A probe located at position 7

    ... | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | ...
                    [ p | p | p | p  ]
                      ^   ^   ^   ^
                      |   | ^ |   |
                      ------|------- A probe located at 7 with width 4 is placed here.
                            |
                            -- The center of the probe is located at 7 + (4 / 2).

Conversion to and from real units
---------------------------------

Conversion from pixel units to real units (e.g. meters) is a scalar
multiplication by the real width of a pixel.

::

    Equation 1: Conversion from pixels to real units

    (distance in pixels) (real units)  = (distance in real units)
                         -------------
                        (per one pixel)

The forbidden edge and minimum field of view width
--------------------------------------------------

Tike does not allow probes to occupy pixels within 1 pixel of the edge of the
field of view. This is to prevent accidentally accessing pixels out of the
field of view (where the values are undefined) when scan positions are
non-integer coordinates. In other words, the minimum coordinate (floor) must be
>= 1 and the maximum coordinate must be <= field of view width - probe width -
1.

::

    Figure 3: The minimum field of view width

    [ x | 1 | 2 | 3 | 4 | x ]
      ^ [ p | p | p | p ] ^
      |   ^   ^   ^   ^   |
      ----|---|---|---|----- These pixels are forbidden because they are the edge pixel
          |   |   |   |
          -------------- These pixels are occupied by a probe of width 4

Thus for a probe with of 4, the smallest field of view is 6 pixels wide and the
only valid probe position is 1. The minimum valid coordinate is 1, and the
maximum valid coordinate is 6 - 4 - 1 = 1. In practice, this means that you
will shift your scan positions so the minimum position is 1. And the minimum
field of view width is the span of the scan positions + probe width + 2.

::

    Equation 2: Minimum field of view width equation

    (minimum field of view width) = (maximum scan position) - (minimum scan position) + (probe width) + 2

Example in one dimension
------------------------

The following python pseudo code would prepare you scan positions for tike and
initialize the minimum-size field of view

.. code-block:: python
    :linenos:

    import numpy as np

    scan_in_pixels = scan_in_meters * pixels_per_meter

    minimum_fov_width = np.amax(scan_in_pixels) - np.amin(scan_in_pixels) +
    probe_width + 2 field_of_view = np.zeros(shape=(minimum_fov_width),
    dtype=np.cfloat)

    scan_in_pixels = scan_in_pixels - np.amin(scan_in_pixels) + 1

Note that :py:func:`tike.ptycho.object.get_padded_object` will do lines 5-8 for you.

"""

from __future__ import annotations
import dataclasses
import logging
import typing
import copy

import cupy as cp
import cupyx.scipy.ndimage
import numpy as np

import tike.communicators
import tike.linalg
import tike.opt
import tike.random
import tike.precision

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class AffineTransform:
    """Represents a 2D affine transformation."""

    scale0: float = 1.0
    scale1: float = 1.0
    shear1: float = 0.0
    angle: float = 0.0
    t0: float = 0.0
    t1: float = 0.0

    def resample(self, factor: float) -> AffineTransform:
        return AffineTransform(
            self.scale0,
            self.scale1,
            self.shear1,
            self.angle,
            self.t0 * factor,
            self.t1 * factor,
        )

    @classmethod
    def fromarray(self, T: np.ndarray) -> AffineTransform:
        """Return an Affine Transfrom from a 2x2 matrix.

        Use decomposition method from Graphics Gems 2 Section 7.1
        """
        xp = cp.get_array_module(T)
        R = T[:2, :2].copy()
        scale0 = xp.linalg.norm(R[0])
        if scale0 <= 0:
            return AffineTransform()
        R[0] /= scale0
        shear1 = R[0] @ R[1]
        R[1] -= shear1 * R[0]
        scale1 = xp.linalg.norm(R[1])
        if scale1 <= 0:
            return AffineTransform()
        R[1] /= scale1
        shear1 /= scale1
        angle = xp.arccos(R[0, 0])
        return AffineTransform(
            scale0=float(scale0),
            scale1=float(scale1),
            shear1=float(shear1),
            angle=float(angle),
            t0=T[2, 0],
            t1=T[2, 1],
        )

    def asarray(self, xp=np) -> np.ndarray:
        """Return an 2x2 matrix of scale, shear, rotation.

        This matrix is scale @ shear @ rotate from left to right.
        """
        cosx = xp.cos(self.angle)
        sinx = xp.sin(self.angle)
        return xp.array(
            [
                [self.scale0, 0.0],
                [0.0, self.scale1],
            ],
            dtype=tike.precision.floating,
        ) @ xp.array(
            [
                [1.0, 0.0],
                [self.shear1, 1.0],
            ],
            dtype=tike.precision.floating,
        ) @ xp.array(
            [
                [+cosx, -sinx],
                [+sinx, +cosx],
            ],
            dtype=tike.precision.floating,
        )

    def asarray3(self, xp=np) -> np.ndarray:
        """Return an 3x2 matrix of scale, shear, rotation, translation.

        This matrix is scale @ shear @ rotate from left to right. Expects a
        homogenous (z) coordinate of 1.
        """
        T = xp.empty((3, 2), dtype=tike.precision.floating)
        T[2] = (self.t0, self.t1)
        T[:2, :2] = self.asarray(xp)
        return T

    def astuple(self) -> tuple:
        """Return the constructor parameters in a tuple."""
        return (
            self.scale0,
            self.scale1,
            self.shear1,
            self.angle,
            self.t0,
            self.t1,
        )

    def __call__(self, x: np.ndarray, gpu=False) -> np.ndarray:
        xp = cp.get_array_module(x)
        return (x @ self.asarray(xp)) + xp.array((self.t0, self.t1))


def estimate_global_transformation(
    positions0: np.ndarray,
    positions1: np.ndarray,
    weights: np.ndarray,
    transform=None,
) -> tuple[AffineTransform, float]:
    """Use weighted least squares to estimate the global affine transformation."""
    xp = cp.get_array_module(positions0)
    try:
        result = AffineTransform.fromarray(
            tike.linalg.lstsq(
                a=xp.pad(positions0, ((0, 0), (0, 1)), constant_values=1),
                b=positions1,
                weights=weights,
            ))
    except np.linalg.LinAlgError:
        # Catch singular matrix when the positions are colinear
        result = AffineTransform()
    return result, np.linalg.norm(result(positions0) - positions1)


def estimate_global_transformation_ransac(
    positions0: np.ndarray,
    positions1: np.ndarray,
    weights: np.ndarray = None,
    transform: AffineTransform = AffineTransform(),
    min_sample: int = 4,  # must be 4 because we are solving for a 2x2 matrix
    max_error: float = 32,
    min_consensus: float = 0.75,
    max_iter: int = 20,
) -> tuple[AffineTransform, float]:
    """Use RANSAC to estimate the global affine transformation.

    Parameters
    ----------
    min_sample
        The number of positions to use to initialize each candidate model
    max_error
        The distance from the model which determines inliar/outliar status
    min_consensus
        The proportion of points needed to accept model as consensus.
    """
    best_fitness = np.inf  # small fitness is good
    # Choose a subset
    # FIXME: Use montecarlo sampling to decide when to stop RANSAC instead of minimum consensus
    for subset in tike.random.randomizer_np.choice(
            a=len(positions0),
            size=(max_iter, min_sample),
            replace=True,
    ):
        # Fit to subset
        candidate_model, _ = estimate_global_transformation(
            positions0=positions0[subset],
            positions1=positions1[subset],
            weights=weights,
            transform=transform,
        )
        # Determine inliars and outliars
        position_error = np.linalg.norm(
            candidate_model(positions0) - positions1,
            axis=-1,
        )
        inliars = (position_error <= max_error)
        # Check if consensus reached
        if np.sum(inliars) / len(inliars) >= min_consensus:
            # Refit with consensus inliars
            candidate_model, fitness = estimate_global_transformation(
                positions0=positions0[inliars],
                positions1=positions1[inliars],
                weights=weights,
                transform=candidate_model,
            )
            if fitness < best_fitness:
                best_fitness = fitness
                transform = candidate_model
    return transform, best_fitness


@dataclasses.dataclass
class PositionOptions:
    """Manage data and settings related to position correction."""

    initial_scan: np.array
    """The original scan positions before they were updated using position
    correction."""

    use_adaptive_moment: bool = False
    """Whether AdaM is used to accelerate the position correction updates."""

    vdecay: float = 0.999
    """The proportion of the second moment that is previous second moments."""

    mdecay: float = 0.9
    """The proportion of the first moment that is previous first moments."""

    use_position_regularization: bool = False
    """Whether the positions are constrained to fit a random error plus affine
    error model."""

    update_magnitude_limit: float = 0
    """When set to a positive number, x and y update magnitudes are clipped
    (limited) to this value."""

    transform: AffineTransform = AffineTransform()
    """Global transform of positions."""

    origin: tuple[float, float] = (0, 0)
    """The rotation center of the transformation. This shift is applied to the
    scan positions before computing the global transformation."""

    confidence: np.ndarray = dataclasses.field(
        init=True,
        default_factory=lambda: None,
    )
    """A rating of the confidence of position information around each position."""

    update_start: int = 0
    """Start position updates at this epoch."""

    def __post_init__(self):
        self.initial_scan = self.initial_scan.astype(tike.precision.floating)
        if self.confidence is None:
            self.confidence = np.ones(
                shape=self.initial_scan.shape,
                dtype=tike.precision.floating,
            )
        if self.use_adaptive_moment:
            self._momentum = np.zeros(
                (*self.initial_scan.shape[:-1], 4),
                dtype=tike.precision.floating,
            )

    def append(self, new_scan):
        self.initial_scan = np.append(
            self.initial_scan,
            values=new_scan,
            axis=-2,
        )
        if self.confidence is not None:
            self.confidence = np.pad(
                self.confidence,
                pad_width=(
                    (0, len(new_scan)),
                    (0, 0),
                ),
                mode='constant',
                constant_values=1.0,
            )
        if self.use_adaptive_moment:
            self._momentum = np.pad(
                self._momentum,
                pad_width=(
                    (0, len(new_scan)),
                    (0, 0),
                ),
                mode='constant',
            )

    def empty(self):
        new = PositionOptions(
            np.empty((0, 2)),
            use_adaptive_moment=self.use_adaptive_moment,
            vdecay=self.vdecay,
            mdecay=self.mdecay,
            use_position_regularization=self.use_position_regularization,
            update_magnitude_limit=self.update_magnitude_limit,
            transform=self.transform,
        )
        if self.use_adaptive_moment:
            new._momentum = np.empty((0, 4))
        return new

    def split(self, indices):
        """Split the PositionOption meta-data along indices."""
        new = PositionOptions(
            self.initial_scan[..., indices, :],
            use_adaptive_moment=self.use_adaptive_moment,
            vdecay=self.vdecay,
            mdecay=self.mdecay,
            use_position_regularization=self.use_position_regularization,
            update_magnitude_limit=self.update_magnitude_limit,
            transform=self.transform,
        )
        if self.confidence is not None:
            new.confidence = self.confidence[..., indices, :]
        if self.use_adaptive_moment:
            new._momentum = self._momentum[..., indices, :]
        return new

    def insert(self, other, indices):
        """Replace the PositionOption meta-data with other data."""
        self.initial_scan[..., indices, :] = other.initial_scan
        if self.confidence is not None:
            self.confidence[..., indices, :] = other.confidence
        if self.use_adaptive_moment:
            self._momentum[..., indices, :] = other._momentum
        return self

    def join(self, other, indices):
        """Replace the PositionOption meta-data with other data."""
        len_scan = self.initial_scan.shape[-2]
        max_index = max(indices.max() + 1, len_scan)
        new_initial_scan = np.empty(
            (*self.initial_scan.shape[:-2], max_index, 2),
            dtype=self.initial_scan.dtype,
        )
        new_initial_scan[..., :len_scan, :] = self.initial_scan
        new_initial_scan[..., indices, :] = other.initial_scan
        self.initial_scan = new_initial_scan
        if self.confidence is not None:
            new_confidence = np.empty(
                (*self.initial_scan.shape[:-2], max_index, 2),
                dtype=self.initial_scan.dtype,
            )
            new_confidence[..., :len_scan, :] = self.confidence
            new_confidence[..., indices, :] = other.confidence
            self.confidence = new_confidence
        if self.use_adaptive_moment:
            new_momentum = np.empty(
                (*self.initial_scan.shape[:-2], max_index, 4),
                dtype=self.initial_scan.dtype,
            )
            new_momentum[..., :len_scan, :] = self._momentum
            new_momentum[..., indices, :] = other._momentum
            self._momentum = new_momentum
        return self

    def copy_to_device(self):
        """Copy to the current GPU memory."""
        options = copy.copy(self)
        options.initial_scan = cp.asarray(self.initial_scan)
        if self.confidence is not None:
            options.confidence = cp.asarray(self.confidence)
        if self.use_adaptive_moment:
            options._momentum = cp.asarray(self._momentum)
        return options

    def copy_to_host(self):
        """Copy to the host CPU memory."""
        options = copy.copy(self)
        options.initial_scan = cp.asnumpy(self.initial_scan)
        if self.confidence is not None:
            options.confidence = cp.asnumpy(self.confidence)
        if self.use_adaptive_moment:
            options._momentum = cp.asnumpy(self._momentum)
        return options

    def resample(self, factor: float) -> PositionOptions:
        """Return a new `PositionOptions` with the parameters scaled."""
        new = PositionOptions(
            initial_scan=self.initial_scan * factor,
            use_adaptive_moment=self.use_adaptive_moment,
            vdecay=self.vdecay,
            mdecay=self.mdecay,
            use_position_regularization=self.use_position_regularization,
            update_magnitude_limit=self.update_magnitude_limit,
            transform=self.transform.resample(factor),
            confidence=self.confidence,
        )
        # Momentum reset to zero when grid scale changes
        return new

    @property
    def vx(self):
        return self._momentum[..., 0]

    @vx.setter
    def vx(self, x):
        self._momentum[..., 0] = x

    @property
    def vy(self):
        return self._momentum[..., 1]

    @vy.setter
    def vy(self, x):
        self._momentum[..., 1] = x

    @property
    def mx(self):
        return self._momentum[..., 2]

    @mx.setter
    def mx(self, x):
        self._momentum[..., 2] = x

    @property
    def my(self):
        return self._momentum[..., 3]

    @my.setter
    def my(self, x):
        self._momentum[..., 3] = x

    @property
    def v(self):
        return self._momentum[..., 0:2]

    @v.setter
    def v(self, x):
        self._momentum[..., 0:2] = x

    @property
    def m(self):
        return self._momentum[..., 2:4]

    @m.setter
    def m(self, x):
        self._momentum[..., 2:4] = x


def check_allowed_positions(scan: np.array, psi: np.array, probe_shape: tuple):
    """Check that all positions are within the field of view.

    Raises
    ------
    ValueError
        The field of view must have 2 pixel buffer around the edge. i.e.
        positions must be >= 2 and < the object.shape - 2 - probe.shape. This
        padding is to allow approximating gradients and to provide better
        interpolation near the edges of the field of view.
    """
    int_scan = scan // 1
    min_corner = np.min(int_scan, axis=-2)
    max_corner = np.max(int_scan, axis=-2)
    valid_min_corner = (1, 1)
    valid_max_corner = (psi.shape[-2] - probe_shape[-2] - 1,
                        psi.shape[-1] - probe_shape[-1] - 1)
    if (np.any(min_corner < valid_min_corner)
            or np.any(max_corner > valid_max_corner)):
        raise ValueError(
            "Scan positions must be >= 1 and "
            "scan positions + 1 + probe.shape must be <= psi.shape. "
            "psi may be too small or the scan positions may be scaled wrong. "
            f"The span of scan is {min_corner} to {max_corner}, and "
            f"the shape of psi is {psi.shape}.")


def update_positions_pd(operator, data, psi, probe, scan,
                        dx=-1, step=0.05):  # yapf: disable
    """Update scan positions using the gradient of intensity method.

    Uses the finite difference method to compute the gradient of the farfield
    intensity with respect to position movement in horizontal and vertical
    directions. Then a least squares solver is used to find the position shift
    that will minimize the intensity error for each of the detector pixels.

    Parameters
    ----------
    farplane : array-like complex64
        The current farplane estimate from psi, probe, scan
    dx : float
        The step size used to estimate the gradient

    References
    ----------
    Dwivedi, Priya, A.P. Konijnenberg, S.F. Pereira, and H.P. Urbach. 2018.
    “Lateral Position Correction in Ptychography Using the Gradient of
    Intensity Patterns.” Ultramicroscopy 192 (September): 29–36.
    https://doi.org/10.1016/j.ultramic.2018.04.004.
    """
    # step 1: the difference between measured and estimate intensity
    intensity, _ = operator._compute_intensity(data, psi, scan, probe)
    dI = (data - intensity).reshape(*data.shape[:-2], np.prod(data.shape[-2:]))

    dI_dx, dI_dy = 0, 0
    for m in range(probe.shape[-3]):

        # step 2: the partial derivatives of wavefront respect to position
        farplane = operator.fwd(psi=psi,
                                scan=scan,
                                probe=probe[..., m:m + 1, :, :])
        dfarplane_dx = (farplane - operator.fwd(
            psi=psi,
            probe=probe[..., m:m + 1, :, :],
            scan=scan + operator.xp.array(
                (0, dx), dtype=tike.precision.floating),
        )) / dx
        dfarplane_dy = (farplane - operator.fwd(
            psi=psi,
            probe=probe[..., m:m + 1, :, :],
            scan=scan + operator.xp.array(
                (dx, 0), dtype=tike.precision.floating),
        )) / dx

        # step 3: the partial derivatives of intensity respect to position
        dI_dx += 2 * np.real(dfarplane_dx * farplane.conj()).reshape(
            *data.shape[:2], -1, *data.shape[2:])

        dI_dy += 2 * np.real(dfarplane_dy * farplane.conj()).reshape(
            *data.shape[:2], -1, *data.shape[2:])

    # step 4: solve for ΔX, ΔY using least squares
    dI_dxdy = np.stack((dI_dy.reshape(*dI.shape), dI_dx.reshape(*dI.shape)),
                       axis=-1)

    grad = tike.linalg.lstsq(a=dI_dxdy, b=dI[..., None])[..., 0]

    logger.debug('grad max: %+12.5e min: %+12.5e', np.max(grad), np.min(grad))
    logger.debug('step size: %3.2g', step)

    # Prevent position drift by keeping center of mass stationary
    center0 = np.mean(scan, axis=-2, keepdims=True)
    scan = scan - step * grad
    center1 = np.mean(scan, axis=-2, keepdims=True)
    scan = scan + (center0 - center1)

    check_allowed_positions(scan, psi, probe.shape)
    cost = operator.cost(data=data, psi=psi, scan=scan, probe=probe).get()
    logger.debug('%10s cost is %+12.5e', 'position', cost)
    return scan, cost


def _gaussian_frequency(sigma, size):
    """Return a gaussian filter in frequency space."""
    arr = cp.fft.fftfreq(size)
    arr *= arr
    scale = sigma * sigma / -2
    arr *= (4 * cp.pi * cp.pi) * scale
    cp.exp(arr, out=arr)
    return arr


def _affine_position_helper(
    scan,
    position_options: PositionOptions,
    max_error,
    relax=0.1,
):
    predicted_positions = position_options.transform(
        position_options.initial_scan)
    err = predicted_positions - position_options.initial_scan
    # constrain more the probes in flat regions
    W = relax * (1 - (position_options.confidence /
                      (1 + position_options.confidence)))
    # penalize positions that are further than max_error from origin; avoid travel larger than max error
    W = cp.minimum(10 * relax,
                   W + cp.maximum(0, err - max_error)**2 / max_error**2)
    # allow free movement in depenence on realibility and max allowed error
    new_scan = scan * (1 - W) + W * predicted_positions
    return new_scan


# TODO: What is a good default value for max_error?
def affine_position_regularization(
    comm: tike.communicators.Comm,
    updated: typing.List[cp.ndarray],
    position_options: typing.List[PositionOptions],
    max_error: float = 32,
    regularization_enabled: bool = False,
) -> typing.Tuple[typing.List[cp.ndarray], typing.List[PositionOptions]]:
    """Regularize position updates with an affine deformation constraint.

    Assume that the true position updates are a global affine transformation
    plus some random error. The regularized positions are then weighted average
    of the affine deformation applied to the original positions and the updated
    positions.

    Parameters
    ----------
    original (..., N, 2)
        The original scanning positions.
    updated (..., N, 2)
        The updated scanning positions.

    Returns
    -------
    regularized (..., N, 2)
        The updated scanning regularized with affine deformation.

    """
    # Gather all of the scanning positions on one host
    positions0 = comm.pool.gather_host(
        [x.initial_scan for x in position_options], axis=0)
    positions1 = comm.pool.gather_host(updated, axis=0)
    positions0 = comm.mpi.Gather(positions0, axis=0, root=0)
    positions1 = comm.mpi.Gather(positions1, axis=0, root=0)

    if comm.mpi.rank == 0:
        new_transform, _ = estimate_global_transformation_ransac(
            positions0=positions0 - position_options[0].origin,
            positions1=positions1 - position_options[0].origin,
            transform=position_options[0].transform,
            max_error=max_error,
        )
    else:
        new_transform = None

    new_transform = comm.mpi.bcast(new_transform, root=0)

    for i in range(len(position_options)):
        position_options[i].transform = new_transform

    if regularization_enabled:
        updated = comm.pool.map(
            _affine_position_helper,
            updated,
            position_options,
            max_error=max_error,
        )

    return updated, position_options


def gaussian_gradient(
    x: cp.ndarray,
    sigma: float = 0.333,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Return 1st order Gaussian derivatives of the last two dimensions of x.

    Don't use scipy.ndimage.gaussian_filter because we only want derivatives
    along last two, not all dimensions.

    References
    ----------
    https://www.crisluengo.net/archives/22/

    """
    return (
        cupyx.scipy.ndimage.gaussian_filter1d(
            -x,
            sigma=sigma,
            order=1,
            axis=-2,
            mode='nearest',
        ),
        cupyx.scipy.ndimage.gaussian_filter1d(
            -x,
            sigma=sigma,
            order=1,
            axis=-1,
            mode='nearest',
        ),
    )
