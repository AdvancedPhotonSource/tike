import numpy as np

from .operator import Operator
from .usfft import eq2us, us2eq


class Lamino(Operator):
    """A Laminography operator.

    Laminography operators to simulate propagation of the beam through the object for a defined tilt angle.

    Attributes
    ----------
    n : int
        The pixel width of the cubic reconstructed grid.
    theta : float32
        Projection angles in the laminography data.    
    tilt : float
        Tilt angle in the laminography setup.

    Parameters
    ----------
    u : (n, n, n) complex64
        The complex refractive index of the object.    
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

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free memory."""

    def fwd(self, u, **kwargs):
        """Perform the forward Laminography transform."""
        # USFFT from equally-spaced grid to unequally-spaced grid
        F = eq2us(u, self.xi, self.n, self.eps, self.xp).reshape(
            [self.ntheta, self.n, self.n])
        
        # Inverse 2D FFT
        data = self.xp.fft.fftshift(self.xp.fft.ifft2(self.xp.fft.fftshift(
            F, axes=(1, 2)), axes=(1, 2), norm="ortho"), axes=(1, 2))
        return data

    def adj(self, data, **kwargs):
        """Perform the adjoint Laminography transform."""
        # Forward 2D FFT
        F = self.xp.fft.fftshift(self.xp.fft.fft2(self.xp.fft.fftshift(
            data, axes=(1, 2)), axes=(1, 2), norm="ortho"), axes=(1, 2)).flatten()
        # Inverse (x->-x) USFFT from unequally-spaced grid to equally-spaced grid
        u = us2eq(F, -self.xi, self.n, self.eps, self.xp)
        return u

    def cost(self, data, obj):
        "Cost function for the least-squres laminography problem"
        return self.xp.linalg.norm((self.fwd(obj)-data).ravel())**2

    def grad(self, data, obj):
        "Gradient for the least-squares laminography problem"
        return self.adj(data=self.fwd(obj) - data) / (self.ntheta * self.n**3)

    def _make_grids(self, theta):
        """Return (ntheta*n*n, 3) unequally-spaced frequencies for the USFFT."""
        [ku, kv] = self.xp.mgrid[-self.n//2: self.n//2, -self.n//2: self.n//2]/self.n
        ku = ku.ravel().astype('float32')
        kv = kv.ravel().astype('float32')
        xi = self.xp.zeros([self.ntheta, self.n*self.n, 3],
                           dtype='float32')
        for itheta in range(self.ntheta):
            xi[itheta, :, 0] = ku*self.xp.cos(theta[itheta]) + \
                kv*self.xp.sin(theta[itheta])*self.xp.cos(self.tilt)
            xi[itheta, :, 1] = ku*self.xp.sin(theta[itheta]) - \
                kv*self.xp.cos(theta[itheta])*self.xp.cos(self.tilt)
            xi[itheta, :, 2] = kv*self.xp.sin(self.tilt)
        # make sure coordinates are in (-0.5,0.5), probably unnecessary
        xi[xi >= 0.5] = 0.5-1e-5
        xi[xi < -0.5] = -0.5+1e-5

        return xi.reshape(self.ntheta*self.n*self.n, 3)
