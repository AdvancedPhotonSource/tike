"""Functions for manipulating and updating scanning positions."""

import logging

import numpy as np
from skimage.feature import register_translation

import tike.linalg

logger = logging.getLogger(__name__)


def update_positionsn(self,
                      scan,
                      target,
                      source,
                      probe,
                      psi,
                      sample=64,
                      beta=100):
    """Update scan positions by comparing previous iteration nearpalne patches."""
    update = np.empty_like(scan)
    for angle in range(target.shape[0]):
        for i in range(target.shape[1]):
            update[angle, i] = register_translation(
                src_image=source[angle, i, 0],
                target_image=target[angle, i, 0],
                upsample_factor=sample,
                return_error=False,
            )
    scan = beta * update + scan
    return scan, None


def update_positionso(self, scan, nearplane, psi, probe, sample=64, beta=100):
    """Update scan positions by comparing previous iteration object patches."""
    source = self.diffraction.fwd(psi=psi, scan=scan)
    grad = (np.conj(probe[:, :, 0]) * (nearplane[:, :, 0] - source) /
            np.square(np.max(np.abs(probe[:, :, 0]))))
    target = source + grad
    update = np.empty_like(scan)
    for angle in range(target.shape[0]):
        for i in range(target.shape[1]):
            update[angle, i], _, _ = register_translation(
                src_image=source[angle, i],
                target_image=target[angle, i],
                upsample_factor=sample,
            )
    scan = beta * update + scan
    return scan, None


def check_allowed_positions(scan, psi, probe_shape):
    """Check that all positions are within the field of view.

    The field of view must have 1 pixel buffer around the edge. i.e. positions
    must be >= 1 and < the object shape - 1 - probe.shape. This padding is to
    allow approximating gradients and to provide better interpolation near the
    edges of the field of view.
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
        raise ValueError("These scan positions exist outside field of view:\n"
                         f"{scan[np.logical_or(x[..., 0], x[..., 1])]}")


def get_padded_object(scan, probe):
    """Return a ones-initialized object and shifted scan positions.

    An complex object array is initialized with shape such that the area
    covered by the probe is padded on each edge by a full probe width. The scan
    positions are shifted to be centered in this newly initialized object
    array.
    """
    # Shift scan positions to zeros
    scan[..., 0] -= np.min(scan[..., 0])
    scan[..., 1] -= np.min(scan[..., 1])

    # Add padding to scan positions of field-of-view / 8
    span = np.max(scan[..., 0]), np.max(scan[..., 1])
    scan[..., 0] += probe.shape[-2]
    scan[..., 1] += probe.shape[-1]

    ntheta = probe.shape[0]
    height = 3 * probe.shape[-2] + int(span[0])
    width = 3 * probe.shape[-1] + int(span[1])

    return np.ones((ntheta, height, width), dtype='complex64'), scan


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
    intensity = operator._compute_intensity(data, psi, scan, probe)
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

    grad = tike.linalg.lstsq(a=dI_dxdy, b=dI, xp=operator.xp)

    logger.debug('grad max: %+12.5e min: %+12.5e', np.max(grad), np.min(grad))
    logger.debug('step size: %3.2g', step)

    # Prevent position drift by keeping center of mass stationary
    center0 = np.mean(scan, axis=-2, keepdims=True)
    scan = scan - step * grad
    center1 = np.mean(scan, axis=-2, keepdims=True)
    scan = scan + (center0 - center1)

    check_allowed_positions(scan, psi, probe.shape)
    cost = operator.cost(data=data, psi=psi, scan=scan, probe=probe)
    logger.info('%10s cost is %+12.5e', 'position', cost)
    return scan, cost
