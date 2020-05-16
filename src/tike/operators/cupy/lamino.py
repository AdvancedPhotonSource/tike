from importlib_resources import files

import cupy as cp

from tike.operators import numpy
from .operator import Operator

_cu_source = files('tike.operators.cupy').joinpath('usfft.cu').read_text()


class Lamino(Operator, numpy.Lamino):

    def __init__(self, *args, **kwargs):
        super(Lamino, self).__init__(
            *args,
            **kwargs,
        )

    def __enter__(self):
        """Return self at start of a with-block."""
        # Call the __enter__ methods for any composed operators.
        # Allocate special memory objects.
        self.scatter_kernel = cp.RawKernel(_cu_source, "scatter")
        self.gather_kernel = cp.RawKernel(_cu_source, "gather")
        return self

    def fwd(self, u, **kwargs):
        """Perform the forward Laminography transform."""

        def gather(xp, Fe, x, n, m, mu):
            return self.gather(Fe, x, n, m, mu)

        # USFFT from equally-spaced grid to unequally-spaced grid
        F = numpy.usfft.eq2us(u,
                              self.xi,
                              self.n,
                              self.eps,
                              self.xp,
                              gather=gather).reshape(
                                  [self.ntheta, self.n, self.n])

        # Inverse 2D FFT
        data = self.xp.fft.fftshift(
            self.xp.fft.ifft2(self.xp.fft.fftshift(F, axes=(1, 2)),
                              axes=(1, 2),
                              norm="ortho"),
            axes=(1, 2),
        )
        return data

    def adj(self, data, **kwargs):
        """Perform the adjoint Laminography transform."""

        def scatter(xp, f, x, n, m, mu):
            return self.scatter(f, x, n, m, mu)

        # Forward 2D FFT
        F = self.xp.fft.fftshift(
            self.xp.fft.fft2(
                self.xp.fft.fftshift(
                    data,
                    axes=(1, 2),
                ),
                axes=(1, 2),
                norm="ortho",
            ),
            axes=(1, 2),
        ).ravel()
        # Inverse (x->-x) USFFT from unequally-spaced grid to equally-spaced
        # grid
        u = numpy.usfft.us2eq(F, -self.xi, self.n, self.eps, self.xp, scatter)
        return u

    def scatter(self, f, x, n, m, mu):
        G = cp.zeros([2 * (n + m)] * 3, dtype="complex64")
        const = cp.array([cp.sqrt(cp.pi / mu)**3, -cp.pi**2 / mu],
                         dtype='float32')
        block = (min(self.scatter_kernel.max_threads_per_block, (2 * m)**3),)
        grid = (1, 0, f.shape[0])
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
        grid = (1, 0, x.shape[0])
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
