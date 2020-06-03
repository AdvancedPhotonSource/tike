import numpy as np

from .operator import Operator
from .usfft import eq2us, us2eq, checkerboard


class Lamino(Operator):
    """A Laminography operator.

    Laminography operators to simulate propagation of the beam through the
    object for a defined tilt angle. An object rotates around its own vertical
    axis, nz, and the beam illuminates the object some tilt angle off this
    axis.

    Attributes
    ----------
    n : int
        The pixel width of the cubic reconstructed grid.
    theta : array-like float32
        The projection angles; rotation around the vertical axis of the object.
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
    """

    def __init__(self, n, theta, tilt, eps=1e-3,
                 **kwargs):  # noqa: D102 yapf: disable
        """Please see help(Lamino) for more info."""
        self.n = n
        self.ntheta = len(theta)
        self.tilt = tilt
        self.eps = eps
        self.xi = self._make_grids(theta)

    def fwd(self, u, **kwargs):
        """Perform the forward Laminography transform."""
        # USFFT from equally-spaced grid to unequally-spaced grid
        F = eq2us(u, self.xi, self.n, self.eps,
                  self.xp).reshape([self.ntheta, self.n, self.n])

        # Inverse 2D FFT
        data = checkerboard(
            self.xp,
            self.xp.fft.ifft2(
                checkerboard(self.xp, F, axes=(1, 2)),
                axes=(1, 2),
                norm="ortho",
            ),
            axes=(1, 2),
            inverse=True,
        )
        return data.astype('complex64')

    def adj(self, data, **kwargs):
        """Perform the adjoint Laminography transform."""
        # Forward 2D FFT
        F = checkerboard(
            self.xp,
            self.xp.fft.fft2(
                checkerboard(
                    self.xp,
                    data.copy(),
                    axes=(1, 2),
                ),
                axes=(1, 2),
                norm="ortho",
            ),
            axes=(1, 2),
            inverse=True,
        ).ravel()
        # Inverse (x->-x) USFFT from unequally-spaced grid to equally-spaced
        # grid
        u = us2eq(F, -self.xi, self.n, self.eps, self.xp)
        return u.astype('complex64')

    def cost(self, data, obj):
        "Cost function for the least-squres laminography problem"
        return self.xp.linalg.norm((self.fwd(obj) - data).ravel())**2

    def grad(self, data, obj):
        "Gradient for the least-squares laminography problem"
        return self.adj(data=self.fwd(obj) - data) / (self.ntheta * self.n**3)

    def _make_grids(self, theta):
        """Return (ntheta*n*n, 3) unequally-spaced frequencies for the USFFT."""
        [kv, ku] = self.xp.mgrid[-self.n // 2:self.n // 2,
                                 -self.n // 2:self.n // 2] / self.n
        ku = ku.ravel().astype('float32')
        kv = kv.ravel().astype('float32')
        xi = self.xp.zeros([self.ntheta, self.n * self.n, 3], dtype='float32')
        ctilt, stilt = self.xp.cos(self.tilt), self.xp.sin(self.tilt)
        for itheta in range(self.ntheta):
            ctheta = self.xp.cos(theta[itheta])
            stheta = self.xp.sin(theta[itheta])
            xi[itheta, :, 2] = ku * ctheta + kv * stheta * ctilt
            xi[itheta, :, 1] = -ku * stheta + kv * ctheta * ctilt
            xi[itheta, :, 0] = kv * stilt
        # make sure coordinates are in (-0.5,0.5), probably unnecessary
        xi[xi >= 0.5] = 0.5 - 1e-5
        xi[xi < -0.5] = -0.5 + 1e-5

        return xi.reshape(self.ntheta * self.n * self.n, 3)
