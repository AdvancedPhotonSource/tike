"""This module provides Solver implementations for a variety of algorithms."""

import logging

import numpy as np

from tike.opt import conjugate_gradient

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
    """
    """
    if recover_psi:
        def cost_psi(psi):
            return operator.cost(data, psi, scan, probe)

        def grad_psi(psi):
            return operator.grad(data, psi, scan, probe)

        psi = conjugate_gradient(
            operator.array_module,
            x=psi,
            cost_function=cost_psi,
            grad=grad_psi,
            num_iter=2,
        )

    return {
        'psi': psi,
        'probe': probe,
    }

def divided(
    self,
    data, probe, scan, psi,
    recover_psi=True, recover_probe=True,
    nmodes=1,
    **kwargs
):  # yapf: disable
    """Solve the Ptychography Problem using method from Odstrcil et al (2018).

    References
    ----------
    Michal Odstrcil, Andreas Menzel, and Manuel Guizar-Sicaros. Iteraive
    least-squares solver for generalized maximum-likelihood ptychography. Optics
    Express. 2018.
    """
    xp = self.array_module
    mode_axis = 2

    # Ensure that the mode dimension is used
    probe = probe.reshape(
        (self.ntheta, -1, nmodes, self.probe_shape, self.probe_shape),
    )
    data = data.reshape(
        (self.ntheta, self.nscan, self.detector_shape, self.detector_shape)
    )

    nearplane = xp.expand_dims(
        self.diffraction.fwd(psi=psi, scan=scan),
        axis=mode_axis,
    ) * probe

    farplane = self.propagation.fwd(nearplane)
    farplane = update_phase(self, data, farplane, nmodes=nmodes)
    nearplane = self.propagation.adj(farplane)

    if recover_psi:
        psi = update_object(self, nearplane, probe, scan, psi, nmodes=nmodes)

    if recover_probe:
        probe = update_probe(self, nearplane, probe, scan, psi, nmodes=nmodes)

    return {
        'psi': psi,
        'probe': probe,
    }

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

    farplane = conjugate_gradient(
        self.array_module,
        x=farplane,
        cost_function=cost_function,
        grad=grad,
        num_iter=num_iter,
    )

    # print cost function for sanity check
    if logger.isEnabledFor(logging.INFO):
        logger.info(' farplane cost is %+12.5e', cost_function(farplane))

    return farplane

def update_probe(self, nearplane, probe, scan, psi, nmodes=1, num_iter=1):
    """Solve the nearplane single probe recovery problem."""
    # name the axes
    position_axis, mode_axis = 1, 2
    xp = self.array_module

    probe = probe.reshape(
        (self.ntheta, -1, nmodes, self.probe_shape, self.probe_shape)
    )
    nearplane = nearplane.reshape(
        (self.ntheta, self.nscan, nmodes, self.probe_shape, self.probe_shape)
    )
    obj_patches = xp.expand_dims(
        self.diffraction.fwd(psi=psi, scan=scan),
        axis=mode_axis,
    )

    def cost_function(probe):
        return xp.sum(xp.square(xp.abs(nearplane - probe * obj_patches)))

    def grad(probe):
        return xp.sum(
            xp.conj(-obj_patches) * (nearplane - probe * obj_patches),
            axis=position_axis, keepdims=True,
        ) / self.nscan

    probe = conjugate_gradient(
        self.array_module,
        x=probe,
        cost_function=cost_function,
        grad=grad,
        num_iter=num_iter,
    )

    if logger.isEnabledFor(logging.INFO):
        cost = cost_function(probe)
        logger.info('nearplane cost is             %+12.5e', cost)

    return probe

def update_object(self, nearplane, probe, scan, psi, nmodes=1, num_iter=1):
    """Solve the nearplane object recovery problem."""
    xp = self.array_module
    mode_axis = 2

    _probe = probe.reshape(
        (self.ntheta, -1, nmodes, self.probe_shape, self.probe_shape)
    )
    _nearplane = nearplane.reshape(
        (self.ntheta, self.nscan, nmodes, self.probe_shape, self.probe_shape)
    )

    for i in range(nmodes):
        nearplane, probe = _nearplane[:, :, i], _probe[:, :, i]

        def cost_function(psi):
            return xp.sum(xp.square(xp.abs(
                _nearplane
                - _probe * xp.expand_dims(
                    self.diffraction.fwd(psi=psi, scan=scan),
                    axis=mode_axis,
                )
            )))

        def grad(psi):
            return self.diffraction.adj(
                xp.conj(-probe)
                * (nearplane - probe * self.diffraction.fwd(psi=psi, scan=scan)),
                scan=scan,
            )

        psi = conjugate_gradient(
            self.array_module,
            x=psi,
            cost_function=cost_function,
            grad=grad,
            num_iter=num_iter,
        )

    if logger.isEnabledFor(logging.INFO):
        cost = cost_function(psi)
        logger.info('nearplane cost is             %+12.5e', cost)

    return psi

def orthogonalize_gs(xp, x):
    """Gram-schmidt orthogonalization for complex arrays.

    x : (..., nmodes, :, :) array_like
        The array with modes in the -3 dimension.

    TODO: Possibly a faster implementation would use QR decomposition.
    """

    def inner(x, y, axis=None):
        """Return the complex inner product of x and y along axis."""
        return xp.sum(xp.conj(x) * y, axis=axis, keepdims=True)

    def norm(x, axis=None):
        """Return the complex vector norm of x along axis."""
        return xp.sqrt(inner(x, x, axis=axis))

    # Reshape x into a 2D array
    unflat_shape = x.shape
    nmodes = unflat_shape[-3]
    x_ortho = x.reshape(*unflat_shape[:-2], -1)

    for i in range(1, nmodes):
        u = x_ortho[..., 0:i,   :]
        v = x_ortho[..., i:i+1, :]
        projections = u * inner(u, v, axis=-1) / inner(u, u, axis=-1)
        x_ortho[..., i:i+1, :] -= xp.sum(projections, axis=-2, keepdims=True)

    if __debug__:
        # Test each pair of vectors for orthogonality
        for i in range(nmodes):
            for j in range(i):
                error = abs(inner(x_ortho[..., i:i+1, :],
                                  x_ortho[..., j:j+1, :], axis=-1))
                assert xp.all(error < 1e-5), (
                    f"Some vectors are not orthogonal!, {error}, {error.shape}"
                )

    return x_ortho.reshape(unflat_shape)
