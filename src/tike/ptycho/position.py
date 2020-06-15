import logging

import numpy as np

logger = logging.getLogger(__name__)


def check_allowed_positions(scan, psi, probe):
    """Check that all positions are within the field of view.

    The field of view must have 1 pixel buffer around the edge. i.e. positions
    must be >= 1 and < the object shape - 1 - probe.shape. This padding is to
    allow approximating gradients and to provide better interpolation near the
    edges of the field of view.
    """
    int_scan = scan // 1
    less_than_one = int_scan < 1
    greater_than_psi = np.stack(
        (int_scan[..., -2] >= psi.shape[-2] - probe.shape[-2],
         int_scan[..., -1] >= psi.shape[-1] - probe.shape[-1]),
        -1,
    )
    if np.any(less_than_one) or np.any(greater_than_psi):
        x = np.logical_or(less_than_one, greater_than_psi)
        raise ValueError("These scan positions exist outside field of view:\n"
                         f"{scan[np.logical_or(x[..., 0], x[..., 1])]}")


def get_padded_object(scan, probe):
    """Return a ones-initialized object and shifted scan positions.

    An complex object array is initialized with shape such that the area
    covered by the probe is padded on each edge by 1/8th the width of that
    dimenion. The scan positions are shifted to be centered in this newly
    initialized object array.
    """
    # Shift scan positions to zeros
    scan[..., 0] -= np.min(scan[..., 0])
    scan[..., 1] -= np.min(scan[..., 1])

    # Add padding to scan positions of field-of-view / 8
    span = np.max(scan[..., 0]), np.max(scan[..., 1])
    scan[..., 0] += span[0] * 0.125
    scan[..., 1] += span[1] * 0.125

    ntheta = probe.shape[0]
    height = probe.shape[-2] + int(span[0] * 1.25)
    width = probe.shape[-1] + int(span[1] * 1.25)

    return np.ones((ntheta, height, width), dtype='complex64'), scan


def _lstsq(a, b, xp):
    """Return the least-squares solution for a @ x = b.

    This implementation, unlike np.linalg.lstsq, allows a stack of matricies to
    be processed simultaneously. The input sizes of the matricies are as
    follows:
        a (..., M, N)
        b (..., M)
        x (...,    N)

    ...seealso:: https://github.com/numpy/numpy/issues/8720
                 https://github.com/cupy/cupy/issues/3062
    """
    assert a.shape[:-1] == b.shape, (f"Leading dims of a {a.shape}"
                                     f"and b {b.shape} must be same!")
    shape = a.shape[:-2]
    a = a.reshape(-1, *a.shape[-2:])
    b = b.reshape(-1, *b.shape[-1:], 1)
    aT = np.swapaxes(a, -1, -2)
    x = xp.empty((a.shape[0], a.shape[-1], 1), dtype=a.dtype)
    for i in range(a.shape[0]):
        x[i] = np.linalg.pinv(aT[i] @ a[i]) @ aT[i] @ b[i]
    return x.reshape(*shape, a.shape[-1])


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

    #update position use the central wavelength only
    i = 1
    for m in range(probe.shape[-3]):

        # step 2: the partial derivatives of wavefront respect to position
        farplane = operator.fwd(psi=psi,
                                scan=scan,
                                probe=probe[..., i:i + 1, m:m + 1, :, :])
        dfarplane_dx = (farplane - operator.fwd(
            psi=psi,
            probe=probe[..., i:i + 1, m:m + 1, :, :],
            scan=scan + operator.xp.array((0, dx), dtype='float32'),
        )) / dx
        dfarplane_dy = (farplane - operator.fwd(
            psi=psi,
            probe=probe[..., i:i + 1, m:m + 1, :, :],
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

    grad = _lstsq(a=dI_dxdy, b=dI, xp=operator.xp)

    logger.debug('grad max: %+12.5e min: %+12.5e', np.max(grad), np.min(grad))
    logger.debug('step size: %3.2g', step)

    # Prevent position drift by keeping center of mass stationary
    center0 = np.mean(scan, axis=-2, keepdims=True)
    scan = scan - step * grad
    center1 = np.mean(scan, axis=-2, keepdims=True)
    scan = scan + (center0 - center1)

    check_allowed_positions(scan, psi, probe)
    cost = operator.cost(data=data, psi=psi, scan=scan, probe=probe)
    logger.info('%10s cost is %+12.5e', 'position', cost)
    return scan, cost
