"""Provides unequally-spaced fast fourier transforms (USFFT).

The USFFT, NUFFT, or NFFT is a fast-fourier transform from an uniform domain to
a non-uniform domain or vice-versa. This module provides forward Fourier
transforms for those two cased. The inverser Fourier transforms may be created
by negating the frequencies on the non-uniform grid.
"""
import numpy as np


def _get_kernel(xp, pad, mu):
    """Return the interpolation kernel for the USFFT."""
    xeq = xp.mgrid[-pad:pad, -pad:pad, -pad:pad]
    return xp.exp(-mu * xp.sum(xeq**2, axis=0)).astype('float32')


def vector_gather(xp, Fe, x, n, m, mu):
    """A faster implementation of sequential_gather"""
    cons = [xp.sqrt(xp.pi / mu)**3, -xp.pi**2 / mu]

    def delta(l, i, x):
        return ((l - m + i).astype('float32') / (2 * n) - x)**2

    F = xp.zeros(x.shape[0], dtype="complex64")
    ell = ((2 * n * x) // 1).astype(xp.int32)  # nearest grid to x
    for i0 in range(2 * m):
        delta0 = delta(ell[:, 0], i0, x[:, 0])
        for i1 in range(2 * m):
            delta1 = delta(ell[:, 1], i1, x[:, 1])
            for i2 in range(2 * m):
                delta2 = delta(ell[:, 2], i2, x[:, 2])
                Fkernel = cons[0] * xp.exp(cons[1] * (delta0 + delta1 + delta2))
                F += Fe[n + ell[:, 0] + i0, n + ell[:, 1] + i1,
                        n + ell[:, 2] + i2] * Fkernel
    return F


def sequential_gather(xp, Fe, x, n, m, mu):
    """Gather F from the regular grid.

    Parameters
    ----------
    Fe : (n, n, n) complex64
        The function at equally spaced frequencies.
    x : (N, 3) float32
        The non-uniform frequencies.

    Returns
    -------
    F : (N, ) complex64
        The values at the non-uniform frequencies.
    """
    cons = [xp.sqrt(xp.pi / mu)**3, -xp.pi**2 / mu]
    F = xp.zeros(x.shape[0], dtype="complex64")
    for k in range(x.shape[0]):
        ell0 = xp.int(xp.floor(2 * n * x[k, 0]))
        ell1 = xp.int(xp.floor(2 * n * x[k, 1]))
        ell2 = xp.int(xp.floor(2 * n * x[k, 2]))
        for i0 in range(2 * m):
            for i1 in range(2 * m):
                for i2 in range(2 * m):
                    kera = cons[0] * xp.exp(cons[1] * (
                        + ((ell0 - m + i0) / (2 * n) - x[k, 0])**2
                        + ((ell1 - m + i1) / (2 * n) - x[k, 1])**2
                        + ((ell2 - m + i2) / (2 * n) - x[k, 2])**2
                    ))  # yapf: disable
                    F[k] += Fe[n + ell0 + i0, n + ell1 + i1,
                               n + ell2 + i2] * kera
    return F


def eq2us(f, x, n, eps, xp, gather=vector_gather):
    """USFFT from equally-spaced grid to unequally-spaced grid.

    Parameters
    ----------
    f : (n, n, n) complex64
        The function to be transformed on a regular-grid of size n.
    x : (N, 3)
        The sampled frequencies on unequally-spaced grid.
    eps : float
        The desired relative accuracy of the USFFT.
    """
    ndim = f.ndim
    pad = n // 2  # where zero-padding stops
    end = pad + n  # where f stops

    # parameters for the USFFT transform
    mu = -xp.log(eps) / (2 * n**2)
    Te = 1 / xp.pi * xp.sqrt(-mu * xp.log(eps) + (mu * n)**2 / 4)
    m = xp.int(xp.ceil(2 * n * Te))

    # smearing kernel (kernel)
    kernel = _get_kernel(xp, pad, mu)

    # FFT and compesantion for smearing
    fe = xp.zeros([2 * n] * ndim, dtype="complex64")
    fe[pad:end, pad:end, pad:end] = f / ((2 * n)**ndim * kernel)
    Fe0 = xp.fft.fftshift(xp.fft.fftn(xp.fft.fftshift(fe)))
    Fe = xp.pad(Fe0, m, mode='wrap')

    F = gather(xp, Fe, x, n, m, mu)

    return F


def sequential_scatter(xp, f, x, n, m, mu):
    """Scatter f to the regular grid.

    Parameters
    ----------
    f : (N, ) complex64
        Values at non-uniform frequencies.
    x : (N, 3) float32
        Non-uniform frequencies.

    Return
    ------
    G : [2 * (n + m)] * 3 complex64

    """
    cons = [xp.sqrt(xp.pi / mu)**3, -xp.pi**2 / mu]
    G = xp.zeros([2 * (n + m)] * 3, dtype="complex64")
    for k in range(x.shape[0]):
        ell0 = xp.int(xp.floor(2 * n * x[k, 0]))
        ell1 = xp.int(xp.floor(2 * n * x[k, 1]))
        ell2 = xp.int(xp.floor(2 * n * x[k, 2]))
        for i0 in range(2 * m):
            for i1 in range(2 * m):
                for i2 in range(2 * m):
                    Fkernel = cons[0] * xp.exp(cons[1] * (
                        + ((ell0 - m + i0) / (2 * n) - x[k, 0])**2
                        + ((ell1 - m + i1) / (2 * n) - x[k, 1])**2
                        + ((ell2 - m + i2) / (2 * n) - x[k, 2])**2
                    ))  # yapf: disable
                    G[n + ell0 + i0,
                      n + ell1 + i1,
                      n + ell2 + i2] += f[k] * Fkernel  # yapf: disable
    return G


def vector_scatter(xp, f, x, n, m, mu, ndim=3):
    """A faster implemenation of sequential_scatter."""
    cons = [xp.sqrt(xp.pi / mu)**ndim, -xp.pi**2 / mu]

    def delta(l, i, x):
        return ((l - m + i).astype('float32') / (2 * n) - x)**2

    G = xp.zeros([(2 * (n + m))**ndim], dtype="complex64")
    ell = ((2 * n * x) // 1).astype(xp.int32)  # nearest grid to x
    stride = ((2 * (n + m))**2, 2 * (n + m))
    for i0 in range(2 * m):
        delta0 = delta(ell[:, 0], i0, x[:, 0])
        for i1 in range(2 * m):
            delta1 = delta(ell[:, 1], i1, x[:, 1])
            for i2 in range(2 * m):
                delta2 = delta(ell[:, 2], i2, x[:, 2])
                Fkernel = cons[0] * xp.exp(cons[1] * (delta0 + delta1 + delta2))
                ids = (
                    n + ell[:, 2] + i2
                    + stride[1] * (n + ell[:, 1] + i1)
                    + stride[0] * (n + ell[:, 0] + i0)
                )  # yapf: disable
                vals = f * Fkernel
                # accumulate by indexes (with possible index intersections),
                # TODO acceleration of bincount!!
                vals = (xp.bincount(ids, weights=vals.real) +
                        1j * xp.bincount(ids, weights=vals.imag))
                ids = xp.nonzero(vals)[0]
                G[ids] += vals[ids]
    return G.reshape([2 * (n + m)] * ndim)


def us2eq(f, x, n, eps, xp, scatter=vector_scatter):
    """USFFT from unequally-spaced grid to equally-spaced grid.

    Parameters
    ----------
    f : (n**3) complex64
        Values of unequally-spaced function on the grid x
    x : (n**3) float
        The frequencies on the unequally-spaced grid
    n : int
        The size of the equall spaced grid.
    eps : float
        The accuracy of computing USFFT
    scatter : function
        The scatter function to use.
    """
    pad = n // 2  # where zero-padding stops
    end = pad + n  # where f stops

    # parameters for the USFFT transform
    mu = -xp.log(eps) / (2 * n**2)
    Te = 1 / xp.pi * xp.sqrt(-mu * xp.log(eps) + (mu * n)**2 / 4)
    m = xp.int(xp.ceil(2 * n * Te))

    # smearing kernel (ker)
    kernel = _get_kernel(xp, pad, mu)

    G = scatter(xp, f, x, n, m, mu)
    G = _unpad(G, m)

    # FFT and compesantion for smearing
    F = xp.fft.fftshift(xp.fft.fftn(xp.fft.fftshift(G)))
    F = F[pad:end, pad:end, pad:end] / ((2 * n)**3 * kernel)

    return F


def _unpad(array, width, mode='wrap'):
    """Remove padding from an array in-place.

    Parameters
    ----------
    array : array
        The array to strip.
    width : int
        The number of indices to remove from both sides along each dimension.
    mode : string
        'wrap' - Add the discarded regions to the array by wrapping them. The
        end regions are added to the beginning and the beginning regions are
        added the end of the new array.

    Returns
    -------
    array : array
        A view of the original array.
    """
    twice = 2 * width
    for _ in range(array.ndim):
        array[+width:+twice] += array[-width:]
        array[-twice:-width] += array[:width]
        array = array[width:-width]
        array = np.moveaxis(array, 0, -1)
    return array
