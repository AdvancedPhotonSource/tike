import logging

import numba
import numpy as np
from skimage.feature import register_translation

logger = logging.getLogger(__name__)


def update_positionsn(self, scan, target, source, probe,  psi, sample=64, beta=100):
    """Update scan positions by comparing previous iteration nearpalne patches."""
    update = np.empty_like(scan)
    for angle in range(target.shape[0]):
        for i in range(target.shape[1]):
            update[angle, i] = register_translation(
                src_image=   source[angle, i, 0],
                target_image=target[angle, i, 0],
                upsample_factor=sample,
                return_error=False,
            )
    scan = beta * update + scan
    return scan, None

def update_positionso(self, scan, nearplane, psi, probe, sample=64, beta=100):
    """Update scan positions by comparing previous iteration object patches."""
    source = self.diffraction.fwd(psi=psi, scan=scan)
    grad = (
        np.conj(probe[:, :, 0]) * (nearplane[:, :, 0] - source)
        / np.square(np.max(np.abs(probe[:, :, 0])))
    )
    target = source + grad
    update = np.empty_like(scan)
    for angle in range(target.shape[0]):
        for i in range(target.shape[1]):
            update[angle, i], _, _ = register_translation(
                src_image=   source[angle, i],
                target_image=target[angle, i],
                upsample_factor=sample,
            )
    scan = beta * update + scan
    return scan, None


def lstsq(a, b):
    """Return the least-squares solution for a @ x = b.

    This implementation, unlike np.linalg.lstsq, allows a stack of matricies to
    be processed simultaneously. The input sizes of the matricies are as
    follows:
        a (..., M, N)
        b (..., M)
        x (...,    N)

    ...seealso:: https://github.com/numpy/numpy/issues/8720
    """
    assert a.shape[:-1] == b.shape, (f"Leading dims of a {a.shape}"
                                     f"and b {b.shape} must be same!")
    shape = a.shape[:-2]
    a = a.reshape(-1, *a.shape[-2:])
    b = b.reshape(-1, *b.shape[-1:], 1)
    aT = np.swapaxes(a, -1, -2)
    x = np.linalg.pinv(aT @ a) @ aT @ b
    return x.reshape(*shape, a.shape[-1])


def update_positions_pd(operator, data, psi, probe, scan,
                        dx=1, step=0.05): # yapf: disable
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
    farplane = operator.fwd(psi=psi, scan=scan, probe=probe)
    intensity = np.square(
        np.linalg.norm(
            farplane.reshape(*data.shape[:2], -1, *data.shape[2:]),
            ord=2,
            axis=2,
        ))
    dI = (data - intensity).reshape(*data.shape[:-2], np.prod(data.shape[-2:]))

    # step 2: the partial derivatives of wavefront respect to position
    dfarplane_dx = (farplane - operator.fwd(
            psi=psi,
        probe=probe,
        scan=scan + np.array((0, dx), dtype='float32'),
    )) / dx
    dfarplane_dy = (farplane - operator.fwd(
            psi=psi,
        probe=probe,
        scan=scan + np.array((dx, 0), dtype='float32'),
    )) / dx

    # step 3: the partial derivatives of intensity respect to position
    # TODO: Find actual deriviatives for multi-mode situation
    dI_dx = 2 * np.sum(np.real(dfarplane_dx * np.conj(farplane)), axis=(-3, -4))
    dI_dy = 2 * np.sum(np.real(dfarplane_dy * np.conj(farplane)), axis=(-3, -4))

    # step 4: solve for ΔX, ΔY using least squares
    dI_dxdy = np.stack((dI_dy.reshape(*dI.shape), dI_dx.reshape(*dI.shape)),
                       axis=-1)

    grad = lstsq(a=dI_dxdy, b=dI)

    logger.info('%10s grad max %+12.5e min %+12.5e', 'position', np.max(grad),
                np.min(grad))

    scan = scan - step * grad
    cost = operator.cost(data=data, psi=psi, scan=scan, probe=probe)

    logger.info('%10s cost is %+12.5e', 'position', cost)
    return scan, cost
