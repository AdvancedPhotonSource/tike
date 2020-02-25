"""This module provides algorithms for solving the ptychography problem."""

import logging

import numpy as np

from tike.opt import conjugate_gradient, line_search

__all__ = [
    "combined",
    "divided",
]

logger = logging.getLogger(__name__)

def combined(
    operator,
    data, probe, scan, psi,
    recover_psi=True, recover_probe=True,
    **kwargs,
):  # yapf: disable
    """Solve the ptychography problem using a combined approach.

    .. seealso:: tike.ptycho.divided
    """
    if recover_psi:

        def cost_psi(psi):
            return operator.cost(data, psi, scan, probe)

        def grad_psi(psi):
            return operator.grad(data, psi, scan, probe)

        psi, cost = conjugate_gradient(
            operator.array_module,
            x=psi,
            cost_function=cost_psi,
            grad=grad_psi,
            num_iter=2,
        )

    return {
        'psi': psi,
        'probe': probe,
        'cost': cost,
    }

def divided(
    self,
    data, probe, scan, psi,
    recover_psi=True, recover_probe=False, recover_positions=False,
    nmodes=1,
    **kwargs
):  # yapf: disable
    """Solve near- and farfield- ptychography problems separately.

    References
    ----------
    Michal Odstrcil, Andreas Menzel, and Manuel Guizar-Sicaros. Iteraive
    least-squares solver for generalized maximum-likelihood ptychography. Optics
    Express. 2018.

    .. seealso:: tike.ptycho.combined
    """
    mode_axis = 2

    # Ensure that the mode dimension is used
    probe = probe.reshape(
        (self.ntheta, -1, nmodes, self.probe_shape, self.probe_shape),)
    data = data.reshape(
        (self.ntheta, self.nscan, self.detector_shape, self.detector_shape))

    nearplane = np.expand_dims(
        self.diffraction.fwd(psi=psi, scan=scan),
        axis=mode_axis,
    ) * probe

    farplane = self.propagation.fwd(nearplane)
    farplane, far_cost = update_phase(self, data, farplane,
                                      nmodes=nmodes, num_iter=2)
    nearplane = self.propagation.adj(farplane)

    if recover_psi:
        psi, near_cost = update_object(self, nearplane, probe, scan, psi,
                                       nmodes=nmodes, num_iter=2)

    if recover_probe:
        probe, near_cost = update_probe(self, nearplane, probe, scan, psi,
                                        nmodes=nmodes, num_iter=2)

    if recover_positions:
        scan, near_cost = update_positions(self, nearplane, psi, probe, scan)

    return {'psi': psi, 'probe': probe, 'cost': near_cost, 'scan': scan}


def update_phase(self, data, farplane, nmodes=1, num_iter=1):
    """Solve the farplane phase problem.

    Parameters
    ----------
    nmodes : int
        The number of incoherent farplane waves that hit the detector
        simultaneously; the number of waves to sum incoherently.
    """
    mode_axis = -3

    def grad(farplane):
        return self.propagation.grad(data, farplane, mode_axis)

    def cost_function(farplane):
        return self.propagation.cost(data, farplane, mode_axis)

    farplane, cost = conjugate_gradient(
        self.array_module,
        x=farplane,
        cost_function=cost_function,
        grad=grad,
        num_iter=num_iter,
    )

    # print cost function for sanity check
    logger.debug(' farplane cost is %+12.5e', cost)

    return farplane, cost


def update_probe(self, nearplane, probe, scan, psi, nmodes=1, num_iter=1):
    """Solve the nearplane single probe recovery problem."""
    # name the axes
    position_axis, mode_axis = 1, 2

    probe = probe.reshape(
        (self.ntheta, -1, nmodes, self.probe_shape, self.probe_shape))
    nearplane = nearplane.reshape(
        (self.ntheta, self.nscan, nmodes, self.probe_shape, self.probe_shape))
    obj_patches = np.expand_dims(
        self.diffraction.fwd(psi=psi, scan=scan),
        axis=mode_axis,
    )

    def cost_function(probe):
        return np.sum(np.square(np.abs(nearplane - probe * obj_patches)))

    def grad(probe):
        return np.sum(
            np.conj(-obj_patches) * (nearplane - probe * obj_patches),
            axis=position_axis,
            keepdims=True,
        ) / self.nscan

    probe, cost = conjugate_gradient(
        self.array_module,
        x=probe,
        cost_function=cost_function,
        grad=grad,
        num_iter=num_iter,
    )

    logger.debug('    probe cost is             %+12.5e', cost)
    return probe, cost


def update_object(self, nearplane, probe, scan, psi, nmodes=1, num_iter=1):
    """Solve the nearplane object recovery problem."""
    mode_axis = 2

    _probe = probe.reshape(
        (self.ntheta, -1, nmodes, self.probe_shape, self.probe_shape))
    _nearplane = nearplane.reshape(
        (self.ntheta, self.nscan, nmodes, self.probe_shape, self.probe_shape))

    for i in range(nmodes):
        nearplane, probe = _nearplane[:, :, i], _probe[:, :, i]

        def cost_function(psi):
            return np.sum(
                np.square(
                    np.abs(_nearplane - _probe * np.expand_dims(
                        self.diffraction.fwd(psi=psi, scan=scan),
                        axis=mode_axis,
                    ))))

        def grad(psi):
            return self.diffraction.adj(
                np.conj(-probe) *
                (nearplane - probe * self.diffraction.fwd(psi=psi, scan=scan)),
                scan=scan,
            )

        psi, cost = conjugate_gradient(
            self.array_module,
            x=psi,
            cost_function=cost_function,
            grad=grad,
            num_iter=num_iter,
        )

    logger.debug('   object cost is             %+12.5e', cost)
    return psi, cost


def update_positions(self, nearplane0, psi, probe, scan):
    """Update scan positions by comparing previous iteration object patches."""
    mode_axis=2
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
        return np.sum(np.square(np.abs(nearplane - nearplane0)))

    scan = scan + grad
    cost = cost_function(scan)

    logger.debug(' position cost is             %+12.5e', cost)
    return scan, cost

def orthogonalize_gs(np, x):
    """Gram-schmidt orthogonalization for complex arrays.

    x : (..., nmodes, :, :) array_like
        The array with modes in the -3 dimension.

    TODO: Possibly a faster implementation would use QR decomposition.
    """

    def inner(x, y, axis=None):
        """Return the complex inner product of x and y along axis."""
        return np.sum(np.conj(x) * y, axis=axis, keepdims=True)

    def norm(x, axis=None):
        """Return the complex vector norm of x along axis."""
        return np.sqrt(inner(x, x, axis=axis))

    # Reshape x into a 2D array
    unflat_shape = x.shape
    nmodes = unflat_shape[-3]
    x_ortho = x.reshape(*unflat_shape[:-2], -1)

    for i in range(1, nmodes):
        u = x_ortho[..., 0:i,   :]
        v = x_ortho[..., i:i+1, :]
        projections = u * inner(u, v, axis=-1) / inner(u, u, axis=-1)
        x_ortho[..., i:i+1, :] -= np.sum(projections, axis=-2, keepdims=True)

    if __debug__:
        # Test each pair of vectors for orthogonality
        for i in range(nmodes):
            for j in range(i):
                error = abs(inner(x_ortho[..., i:i+1, :],
                                  x_ortho[..., j:j+1, :], axis=-1))
                assert np.all(error < 1e-5), (
                    f"Some vectors are not orthogonal!, {error}, {error.shape}"
                )

    return x_ortho.reshape(unflat_shape)
