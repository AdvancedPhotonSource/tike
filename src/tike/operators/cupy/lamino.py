__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

try:
    from importlib.resources import files
except ImportError:
    # Backport for python<3.9 available as importlib_resources package
    from importlib_resources import files

import cupy as cp
import cupyx.scipy.fft
import numpy as np

from .cache import CachedFFT
from .usfft import eq2us, us2eq, checkerboard
from .operator import Operator

_cu_source = files('tike.operators.cupy').joinpath('grid.cu').read_text()
_make_grids_kernel = cp.RawKernel(_cu_source, "make_grids")


class Lamino(CachedFFT, Operator):
    """A Laminography operator.

    Laminography operators to simulate propagation of the beam through the
    object for a defined tilt angle. An object rotates around its own vertical
    axis, nz, and the beam illuminates the object some tilt angle off this
    axis.

    Attributes
    ----------
    n : int
        The pixel width of the cubic reconstructed grid.
    tilt : float32 [radians]
        The tilt angle; the angle between the rotation axis of the object and
        the light source. π / 2 for conventional tomography. 0 for a beam path
        along the rotation axis.

    Parameters
    ----------
    u : (nz, n, n) complex64
        The complex refractive index of the object. nz is the axis
        corresponding to the rotation axis.
    data : (ntheta, n, n) complex64
        The complex projection data of the object.
    theta : array-like float32 [radians]
        The projection angles; rotation around the vertical axis of the object.
    """

    def __init__(self, n, tilt, eps=1e-3, upsample=1, **kwargs):
        """Please see help(Lamino) for more info."""
        self.n = n
        self.tilt = np.float32(tilt)
        self.eps = np.float32(eps)
        self.upsample = upsample

    def fwd(self, u, theta, **kwargs):
        """Perform the forward Laminography transform."""

        def _fftn(*args, **kwargs):
            return self._fftn(*args, overwrite_x=True, **kwargs)

        xi = self._make_grids(theta)

        # USFFT from equally-spaced grid to unequally-spaced grid
        F = eq2us(
            u,
            xi,
            self.n,
            self.eps,
            self.xp,
            fftn=_fftn,
            upsample=self.upsample,
        ).reshape([theta.shape[-1], self.n, self.n])

        # Inverse 2D FFT
        data = checkerboard(
            self.xp,
            self._ifft2(
                checkerboard(
                    self.xp,
                    F,
                    axes=(1, 2),
                ),
                axes=(1, 2),
                overwrite_x=True,
            ),
            axes=(1, 2),
            inverse=True,
        )
        return data

    def adj(self, data, theta, overwrite=False, **kwargs):
        """Perform the adjoint Laminography transform."""

        def _fftn(*args, **kwargs):
            return self._fftn(*args, overwrite_x=True, **kwargs)

        xi = self._make_grids(theta)

        # Forward 2D FFT
        F = checkerboard(
            self.xp,
            self._fft2(
                checkerboard(
                    self.xp,
                    data.copy() if not overwrite else data,
                    axes=(1, 2),
                ),
                axes=(1, 2),
                overwrite_x=True,
            ),
            axes=(1, 2),
            inverse=True,
        ).ravel()
        # Inverse (x->-x / n**2) USFFT from unequally-spaced grid to
        # equally-spaced grid.
        u = us2eq(
            F,
            -xi,
            self.n,
            self.eps,
            self.xp,
            fftn=_fftn,
            upsample=self.upsample,
        )
        u /= self.n**2
        return u

    def cost(self, data, theta, obj):
        """Cost function for the least-squres laminography problem"""
        return self.xp.linalg.norm((self.fwd(
            u=obj,
            theta=theta,
        ) - data).ravel())**2

    def grad(self, data, theta, obj):
        """Gradient for the least-squares laminography problem"""
        out = self.adj(
            data=self.fwd(
                u=obj,
                theta=theta,
            ) - data,
            theta=theta,
        )
        # BUG? Cannot joint line below and above otherwise types are promoted?
        out /= (data.shape[-3] * self.n**3)
        return out

    def _make_grids(self, theta):
        """Return (ntheta*n*n, 3) unequally-spaced frequencies for the USFFT."""

        assert self.tilt.dtype == np.float32
        assert theta.dtype == cp.float32, theta.dtype

        xi = cp.empty((theta.shape[-1] * self.n * self.n, 3), dtype="float32")

        grid = (
            -(-self.n // _make_grids_kernel.max_threads_per_block),
            self.n,
            theta.shape[-1],
        )
        block = (min(self.n, _make_grids_kernel.max_threads_per_block),)
        _make_grids_kernel(grid, block, (
            xi,
            theta,
            theta.shape[-1],
            self.n,
            self.tilt,
        ))
        return xi
