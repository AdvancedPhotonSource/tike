"""Functions for manipulating and updating scanning positions."""

import dataclasses
import logging

import cupy as cp
import numpy as np

import tike.linalg

logger = logging.getLogger(__name__)


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

    def __post_init__(self):
        if self.use_adaptive_moment:
            self._momentum = np.zeros(
                (*self.initial_scan.shape[:-1], 4),
                dtype='float32',
            )

    def append(self, new_scan):
        self.initial_scan = np.append(
            self.initial_scan,
            values=new_scan,
            axis=-2,
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
        )
        if self.use_adaptive_moment:
            new._momentum = self._momentum[..., indices, :]
        return new

    def insert(self, other, indices):
        """Replace the PositionOption meta-data with other data."""
        self.initial_scan[..., indices, :] = other.initial_scan
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
        self.initial_scan = cp.asarray(self.initial_scan)
        if self.use_adaptive_moment:
            self._momentum = cp.asarray(self._momentum)
        return self

    def copy_to_host(self):
        """Copy to the host CPU memory."""
        self.initial_scan = cp.asnumpy(self.initial_scan)
        if self.use_adaptive_moment:
            self._momentum = cp.asnumpy(self._momentum)
        return self

    def resample(self, factor):
        """Return a new PositionOptions with the parameters scaled."""
        new = PositionOptions(
            initial_scan=self.initial_scan * factor,
            use_adaptive_moment=self.use_adaptive_moment,
            vdecay=self.vdecay,
            mdecay=self.mdecay,
            use_position_regularization=self.use_position_regularization,
        )
        # Momentum reset to zero when grid scale changes

    @property
    def vx(self):
        return self._momentum[..., 0]

    @vx.setter
    def vx(self, x):
        self._momentum[..., 0] = x

    @property
    def vy(self):
        return self._momentum[..., 1]

    @vx.setter
    def vy(self, x):
        self._momentum[..., 1] = x

    @property
    def mx(self):
        return self._momentum[..., 2]

    @vx.setter
    def mx(self, x):
        self._momentum[..., 2] = x

    @property
    def my(self):
        return self._momentum[..., 3]

    @vx.setter
    def my(self, x):
        self._momentum[..., 3] = x


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
    less_than_one = int_scan < 1
    greater_than_psi = np.stack(
        (int_scan[..., -2] >= psi.shape[-2] - probe_shape[-2],
         int_scan[..., -1] >= psi.shape[-1] - probe_shape[-1]),
        -1,
    )
    if np.any(less_than_one) or np.any(greater_than_psi):
        x = np.logical_or(less_than_one, greater_than_psi)
        raise ValueError(
            "Scan positions must be >= 2 and "
            "scan positions + 2 + probe.shape must be < object.shape. "
            "psi may be too small or the scan positions may be scaled wrong. "
            "These scan positions exist outside field of view:\n"
            f"{scan[np.logical_or(x[..., 0], x[..., 1])]}")


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
            scan=scan + operator.xp.array((0, dx), dtype='float32'),
        )) / dx
        dfarplane_dy = (farplane - operator.fwd(
            psi=psi,
            probe=probe[..., m:m + 1, :, :],
            scan=scan + operator.xp.array((dx, 0), dtype='float32'),
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
    logger.info('%10s cost is %+12.5e', 'position', cost)
    return scan, cost


def _image_grad(op, x):
    """Return the gradient of the x for each of the last two dimesions."""
    # FIXME: Use different gradient approximation that does not use FFT. Because
    # FFT caches are per-thread and per-device, using FFT is inefficient.
    ramp = 2j * cp.pi * cp.linspace(
        -0.5,
        0.5,
        x.shape[-1],
        dtype='float32',
        endpoint=False,
    )
    grad_x = op.propagation._ifftn(
        ramp[:, None] * op.propagation._fftn(x, axes=(-2,)),
        axes=(-2,),
    )
    grad_y = op.propagation._ifftn(
        ramp * op.propagation._fftn(x, axes=(-1,)),
        axes=(-1,),
    )
    return grad_x, grad_y


def _gaussian_frequency(sigma, size):
    """Return a gaussian filter in frequency space."""
    arr = cp.fft.fftfreq(size)
    arr *= arr
    scale = sigma * sigma / -2
    arr *= (4 * cp.pi * cp.pi) * scale
    cp.exp(arr, out=arr)
    return arr


# TODO: What is a good default value for max_error?
def affine_position_regularization(
    op,
    psi,
    probe,
    original,
    updated,
    max_error=None,
):
    """Regularize position updates with an affine deformation constraint.

    Assume that the true position updates are a global affine transformation
    plus some random error. The regularized positions are then weighted average
    of the affine deformation applied to the original positions and the updated
    positions.

    The affine deformation, X, is represented as a (..., 2, 2) array such that
    updated = original @ X. X may be decomposed into scale, rotation, and shear
    operations.

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
    transformation (..., 2, 2)
        The global affine transformation

    References
    ----------
    This algorithm copied from ptychoshelves.

    """
    # Estimate the reliability of each updated position based on the content of
    # the patch of the object at that position; smooth patches are less
    # reliable than patches with interesting features. This position relability
    # is some imperical formula based on weighting the local image gradient of
    # the object by the amount of illumination it recieved.

    obj_proj = op.diffraction.patch.fwd(
        images=psi / cp.max(cp.abs(psi), axis=(-1, -2), keepdims=True),
        positions=updated,
        patch_width=probe.shape[-1],
    )
    nx, ny = obj_proj.shape[-2:]
    X, Y = cp.mgrid[-ny // 2:ny // 2, -nx // 2:nx // 2]
    spatial_filter = cp.exp(-(X**16 + Y**16) / (min(nx, ny) / 2.2)**16)
    obj_proj *= spatial_filter
    dX, dY = _image_grad(op, obj_proj)

    illum = probe[..., :, 0, 0, :, :]
    illum = illum * illum.conj()
    illum = cp.tile(illum, (1, updated.shape[-2], 1, 1))
    sigma = probe.shape[-1] / 10
    total_illumination = op.diffraction.patch.adj(
        patches=illum,
        images=cp.zeros(psi.shape, dtype='complex64'),
        positions=updated,
    )
    total_illumination = op.propagation._fft2(total_illumination)
    total_illumination *= _gaussian_frequency(
        sigma=sigma,
        size=total_illumination.shape[-1],
    )
    total_illumination *= _gaussian_frequency(
        sigma=sigma,
        size=total_illumination.shape[-2],
    )[..., None]
    total_illumination = op.propagation._ifft2(total_illumination)
    illum_proj = op.diffraction.patch.fwd(
        images=total_illumination,
        positions=updated,
        patch_width=probe.shape[-1],
    )
    dX = abs(dX) * illum_proj.real * illum.real
    dY = abs(dY) * illum_proj.real * illum.real

    total_variation = np.stack(
        (
            cp.sqrt(cp.mean(dX, axis=(-1, -2))),
            cp.sqrt(cp.mean(dY, axis=(-1, -2))),
        ),
        axis=-1,
    )

    position_reliability = total_variation**4 / cp.mean(
        total_variation**4, axis=-2, keepdims=True)

    # Use weighted least squares to find the global affine transformation, X.
    # The two columns of X are independent; we solve separtely so we can use
    # different weights in each direction.
    # TODO: Use homogenous coordinates to add shifts into model
    X = cp.empty((*updated.shape[:-2], 2, 2), dtype='float32')
    X[..., 0:1] = tike.linalg.lstsq(
        b=updated[..., 0:1],
        a=original,
        weights=position_reliability[..., 0],
    )
    X[..., 1:2] = tike.linalg.lstsq(
        b=updated[..., 1:2],
        a=original,
        weights=position_reliability[..., 1],
    )

    logger.info(f'affine position error:\n{X}')

    # TODO: Decompose X into scale, rotate, shear operations.
    # Remove non-affine and unwanted transformations
    # scale, rotate, shear = _decompose_transformation()
    # X = scale @ rotate @ shear

    # Regularize the positions based on the position reliability and distance
    # from the original positions.
    relax = 0.1
    # Constrain more the probes in flat regions
    W = relax * (1 - (position_reliability / (1 + position_reliability)))
    # Penalize positions with a large random error
    if max_error is not None:
        random_error = updated - original @ X
        W = cp.minimum(
            10 * relax,
            W + cp.maximum(0, random_error - max_error)**2 / max_error**2,
        )
    return (1 - W) * updated + W * original @ X, X
