__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

from importlib_resources import files

import cupy as cp

from .cache import CachedFFT
from .usfft import eq2us, us2eq, checkerboard
from .operator import Operator

_cu_source = files('tike.operators.cupy').joinpath('usfft.cu').read_text()


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
    tilt : float32
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
    theta : array-like float32
        The projection angles; rotation around the vertical axis of the object.
    """

    def __init__(self, n, tilt, eps=1e-3,
                 **kwargs):  # noqa: D102 yapf: disable
        """Please see help(Lamino) for more info."""
        self.n = n
        self.tilt = tilt
        self.eps = eps

    def __enter__(self):
        """Return self at start of a with-block."""
        CachedFFT.__enter__(self)
        # Call the __enter__ methods for any composed operators.
        # Allocate special memory objects.
        self.scatter_kernel = cp.RawKernel(_cu_source, "scatter")
        self.gather_kernel = cp.RawKernel(_cu_source, "gather")
        return self

    def fwd(self, u, theta, **kwargs):
        """Perform the forward Laminography transform."""

        xi = self._make_grids(theta)

        def gather(xp, Fe, x, n, m, mu):
            return self.gather(Fe, x, n, m, mu)

        def fftn(*args, **kwargs):
            return self._fftn(*args, overwrite=True, **kwargs)

        # USFFT from equally-spaced grid to unequally-spaced grid
        F = eq2us(u, xi, self.n, self.eps, self.xp, gather,
                  fftn).reshape([theta.shape[-1], self.n, self.n])

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
                overwrite=True,
            ),
            axes=(1, 2),
            inverse=True,
        )
        return data

    def adj(self, data, theta, overwrite=False, **kwargs):
        """Perform the adjoint Laminography transform."""

        xi = self._make_grids(theta)

        def scatter(xp, f, x, n, m, mu):
            return self.scatter(f, x, n, m, mu)

        def fftn(*args, **kwargs):
            return self._fftn(*args, overwrite=True, **kwargs)

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
                overwrite=True,
            ),
            axes=(1, 2),
            inverse=True,
        ).ravel()
        # Inverse (x->-x) USFFT from unequally-spaced grid to equally-spaced
        # grid
        u = us2eq(F, -xi, self.n, self.eps, self.xp, scatter, fftn)
        u /= self.n**2
        return u

    def scatter(self, f, x, n, m, mu):
        G = cp.zeros([2 * n] * 3, dtype="complex64")
        const = cp.array([cp.sqrt(cp.pi / mu)**3, -cp.pi**2 / mu],
                         dtype='float32')
        block = (min(self.scatter_kernel.max_threads_per_block, (2 * m)**3),)
        grid = (1, 0, min(f.shape[0], 65535))
        self.scatter_kernel(grid, block, (
            G,
            f.astype('complex64'),
            f.shape[0],
            x.astype('float32'),
            n,
            m,
            const.astype('float32'),
        ))
        return G

    def gather(self, Fe, x, n, m, mu):
        F = cp.zeros(x.shape[0], dtype="complex64")
        const = cp.array([cp.sqrt(cp.pi / mu)**3, -cp.pi**2 / mu],
                         dtype='float32')
        block = (min(self.scatter_kernel.max_threads_per_block, (2 * m)**3),)
        grid = (1, 0, min(x.shape[0], 65535))
        self.gather_kernel(grid, block, (
            F,
            Fe.astype('complex64'),
            x.shape[0],
            x.astype('float32'),
            n,
            m,
            const.astype('float32'),
        ))
        return F

    def cost(self, data, theta, obj):
        "Cost function for the least-squres laminography problem"
        return self.xp.linalg.norm((self.fwd(
            u=obj,
            theta=theta,
        ) - data).ravel())**2

    def grad(self, data, theta, obj):
        "Gradient for the least-squares laminography problem"
        return self.adj(
            data=self.fwd(
                u=obj,
                theta=theta,
            ) - data,
            theta=theta,
        ) / (data.shape[-3] * self.n**3)

    def _make_grids(self, theta):
        """Return (ntheta*n*n, 3) unequally-spaced frequencies for the USFFT."""
        [kv, ku] = self.xp.mgrid[-self.n // 2:self.n // 2,
                                 -self.n // 2:self.n // 2] / self.n
        ku = ku.ravel().astype('float32')
        kv = kv.ravel().astype('float32')
        xi = self.xp.zeros([theta.shape[-1], self.n * self.n, 3],
                           dtype='float32')
        ctilt, stilt = self.xp.cos(self.tilt), self.xp.sin(self.tilt)
        for itheta in range(theta.shape[-1]):
            ctheta = self.xp.cos(theta[itheta])
            stheta = self.xp.sin(theta[itheta])
            xi[itheta, :, 2] = ku * ctheta + kv * stheta * ctilt
            xi[itheta, :, 1] = -ku * stheta + kv * ctheta * ctilt
            xi[itheta, :, 0] = kv * stilt
        # make sure coordinates are in (-0.5,0.5), probably unnecessary
        xi[xi >= 0.5] = 0.5 - 1e-5
        xi[xi < -0.5] = -0.5 + 1e-5

        return xi.reshape(theta.shape[-1] * self.n * self.n, 3)
