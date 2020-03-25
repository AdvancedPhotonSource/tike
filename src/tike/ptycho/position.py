import numpy as np
from skimage.feature import register_translation

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

def update_positions(self, nearplane0, psi, probe, scan):
    """Update scan positions by comparing previous iteration object patches."""
    mode_axis = 2
    nmodes = 1

    # Ensure that the mode dimension is used
    probe = probe.reshape(
        (self.ntheta, -1, nmodes, self.probe_shape, self.probe_shape),)
    nearplane0 = nearplane0.reshape(
        (self.ntheta, -1, nmodes, self.probe_shape, self.probe_shape),)

    def least_squares(a, b):
        """Return the least-squares solution for a @ x = b.

        This implementation, unlike np.linalg.lstsq, allows a stack of
        matricies to be processed simultaneously. The input sizes of the
        matricies are as follows:
            a (..., M, N)
            b (..., M)
            x (...,    N)

        ...seealso:: https://github.com/numpy/numpy/issues/8720
        """
        shape = a.shape[:-2]
        a = a.reshape(-1, *a.shape[-2:])
        b = b.reshape(-1, *b.shape[-1:], 1)
        x = np.empty((a.shape[0], a.shape[-1]))
        aT = np.swapaxes(a, -1, -2)
        x = np.linalg.pinv(aT @ a) @ aT @ b
        return x.reshape(*shape, a.shape[-1])

    nearplane = np.expand_dims(
        self.diffraction.fwd(
            psi=psi,
            scan=scan,
        ),
        axis=mode_axis,
    ) * probe

    dn = (nearplane0 - nearplane).view('float32')
    dn = dn.reshape(*dn.shape[:-2], -1)

    ndx = (np.expand_dims(
        self.diffraction.fwd(
            psi=psi,
            scan=scan + np.array((0, 1), dtype='float32'),
        ),
        axis=mode_axis,
    ) * probe - nearplane).view('float32')

    ndy = (np.expand_dims(
        self.diffraction.fwd(
            psi=psi,
            scan=scan + np.array((1, 0), dtype='float32'),
        ),
        axis=mode_axis,
    ) * probe - nearplane).view('float32')

    dxy = np.stack(
        (
            ndy.reshape(*ndy.shape[:-2], -1),
            ndx.reshape(*ndx.shape[:-2], -1),
        ), axis=-1)

    grad = least_squares(a=dxy, b=dn)

    grad = np.mean(grad, axis=mode_axis)

    def cost_function(scan):
        nearplane = np.expand_dims(
            self.diffraction.fwd(
                psi=psi,
                scan=scan,
            ),
            axis=mode_axis,
        ) * probe
        return np.linalg.norm(nearplane - nearplane0)

    step = 0.01
    scan = scan + step * grad
    cost = cost_function(scan)

    logger.debug(' position cost is             %+12.5e', cost)
    return scan, cost
