"""Provides unequally-spaced fast fourier transforms (USFFT).

The USFFT, NUFFT, or NFFT is a fast-fourier transform from an uniform domain to
a non-uniform domain or vice-versa. This module provides forward Fourier
transforms for those two cased. The inverser Fourier transforms may be created
by negating the frequencies on the non-uniform grid.

The implementation of this USFFT is the composition of the following
operations: zero-padding, interpolation-kernel-correction, FFT, and
linear-interpolation.
"""
import cupy as cp
try:
    from importlib.resources import files
except ImportError:
    # Backport for python<3.9 available as importlib_resources package
    from importlib_resources import files

_cu_source = files('tike.operators.cupy').joinpath('usfft.cu').read_text()

_scatter_kernel = cp.RawKernel(_cu_source, "scatter")
_gather_kernel = cp.RawKernel(_cu_source, "gather")


def _get_kernel(xp, n, mu):
    """Return the interpolation kernel for the USFFT."""
    pad = n // 2
    end = n - pad
    u = -mu * xp.arange(-pad, end, dtype='float32')**2
    kernel_shape = (len(u), len(u), len(u))
    norm = xp.zeros(kernel_shape, dtype='float32')
    norm += u
    norm += u[:, None]
    norm += u[:, None, None]
    return xp.exp(norm)


def vector_gather(xp, Fe, x, n, m, mu):
    """Gather F from the regular grid.

    Parameters
    ----------
    Fe : (n, n, n) complex64
        The function at equally spaced frequencies. Frequencies on the grid are
        zero-centered i.e. [ -0.5, 0.25, 0.0,  0.25]
    x : (N, 3) float32
        The non-uniform frequencies in the range [-0.5, 0.5)
    n : int
        The width of Fe along each edge
    m : int
        The width of the interpolation kernel along each edge.

    Returns
    -------
    F : (N, ) complex64
        The values at the non-uniform frequencies.
    """
    cons = [xp.sqrt(xp.pi / mu)**3, -xp.pi**2 / mu]
    half = n // 2

    def delta(ell, i, x):
        return ((ell + i).astype('float32') / n - x)**2

    F = xp.zeros(x.shape[0], dtype="complex64")
    ell = ((n * x) // 1).astype(xp.int32)  # nearest grid to x
    for i0 in range(-m, m):
        delta0 = delta(ell[:, 0], i0, x[:, 0])
        for i1 in range(-m, m):
            delta1 = delta(ell[:, 1], i1, x[:, 1])
            for i2 in range(-m, m):
                delta2 = delta(ell[:, 2], i2, x[:, 2])
                Fkernel = cons[0] * xp.exp(cons[1] * (delta0 + delta1 + delta2))
                F += Fe[(half + ell[:, 0] + i0) % n,
                        (half + ell[:, 1] + i1) % n,
                        (half + ell[:, 2] + i2) % n] * Fkernel
    return F


def gather(_, Fe, x, n, m, mu):
    """See vector_gather documenation."""
    F = cp.zeros(x.shape[0], dtype="complex64")
    const = cp.array([cp.sqrt(cp.pi / mu)**3, -cp.pi**2 / mu], dtype='float32')
    assert F.dtype == cp.complex64
    assert Fe.dtype == cp.complex64
    assert x.dtype == cp.float32
    assert const.dtype == cp.float32
    block = (min(_scatter_kernel.max_threads_per_block, (2 * m)**3),)
    grid = (1, 0, min(x.shape[0], 65535))
    _gather_kernel(grid, block, (
        F,
        Fe,
        x.shape[0],
        x,
        n,
        m,
        const,
    ))
    return F


def eq2us(f, x, n, eps, xp, gather=gather, fftn=None, upsample=2):
    """USFFT from equally-spaced grid to unequally-spaced grid.

    Parameters
    ----------
    f : (n, n, n) complex64
        The function at equally-spaced frequencies. Frequencies on the grid are
        zero-centered i.e. [ -0.5, 0.25, 0.0,  0.25]
    x : (N, 3) float32
        The frequencies on the unequally-spaced grid in the range [-0.5, 0.5)
    n : int
        The size of the equally-spaced grid along each edge.
    eps : float
        The accuracy of computing USFFT.
    upsample : float >= 1
        The ratio of the upsampled grid to the equally-spaced grid.

    Returns
    -------
    F : (N, ) complex64
        Values of unequally-spaced function on the grid x.

    """
    fftn = xp.fft.fftn if fftn is None else fftn
    upsampled = 2 * int(upsample * n / 2)  # upsampled grid is always even-sized
    pad = (upsampled - n) // 2  # where zero-padding stops
    end = pad + n  # where f stops

    # parameters for the USFFT transform
    mu = -xp.log(eps) / (2 * n**2)
    Te = 1 / xp.pi * xp.sqrt(-mu * xp.log(eps) + (mu * n)**2 / 4)
    m = int(xp.ceil(upsampled * Te))

    # smearing kernel (ker)
    kernel = _get_kernel(xp, n, mu)
    kernel *= upsampled**3

    # FFT and compesantion for smearing
    fe = xp.zeros([upsampled] * 3, dtype="complex64")

    fe[pad:end, pad:end, pad:end] = f / kernel
    Fe = checkerboard(xp, fftn(checkerboard(xp, fe)), inverse=True)
    F = gather(xp, Fe, x, upsampled, m, mu)

    return F


def vector_scatter(xp, f, x, n, m, mu, ndim=3):
    """Scatter f to the regular grid.

    Parameters
    ----------
    f : (N, ) complex64
        Values at non-uniform frequencies.
    x : (N, 3) float32
        The non-uniform frequencies in the range [-0.5, 0.5)
    n : int
        The width of G along each edge
    m : int
        The width of the interpolation kernel along each edge.

    Return
    ------
    G : (n, n, n) complex64
        The function at equally spaced frequencies. Frequencies on the grid are
        zero-centered i.e. [ -0.5, 0.25, 0.0,  0.25]

    """
    cons = [xp.sqrt(xp.pi / mu)**ndim, -xp.pi**2 / mu]
    half = n // 2

    def delta(ell, i, x):
        return ((ell + i).astype('float32') / n - x)**2

    G = xp.zeros([n**ndim], dtype="complex64")
    ell = ((n * x) // 1).astype(xp.int32)  # nearest grid to x
    stride = (n**2, n)
    for i0 in range(-m, m):
        delta0 = delta(ell[:, 0], i0, x[:, 0])
        for i1 in range(-m, m):
            delta1 = delta(ell[:, 1], i1, x[:, 1])
            for i2 in range(-m, m):
                delta2 = delta(ell[:, 2], i2, x[:, 2])
                Fkernel = cons[0] * xp.exp(cons[1] * (delta0 + delta1 + delta2))
                ids = (((half + ell[:, 2] + i2) % n) +
                       ((half + ell[:, 1] + i1) % n) * stride[1] +
                       ((half + ell[:, 0] + i0) % n) * stride[0])
                vals = f * Fkernel
                # accumulate by indexes (with possible index intersections),
                # TODO acceleration of bincount!!
                vals = (xp.bincount(ids, weights=vals.real) +
                        1j * xp.bincount(ids, weights=vals.imag))
                ids = xp.nonzero(vals)[0]
                G[ids] += vals[ids]
    return G.reshape([n] * ndim)


def scatter(_, f, x, n, m, mu):
    """See vector_scatter documenation"""
    G = cp.zeros([n] * 3, dtype="complex64")
    const = cp.array([cp.sqrt(cp.pi / mu)**3, -cp.pi**2 / mu], dtype='float32')
    assert G.dtype == cp.complex64
    assert f.dtype == cp.complex64
    assert x.dtype == cp.float32
    assert const.dtype == cp.float32
    block = (min(_scatter_kernel.max_threads_per_block, (2 * m)**3),)
    grid = (1, 0, min(f.shape[0], 65535))
    _scatter_kernel(grid, block, (
        G,
        f,
        f.shape[0],
        x,
        n,
        m,
        const,
    ))
    return G


def us2eq(f, x, n, eps, xp, scatter=scatter, fftn=None, upsample=2):
    """USFFT from unequally-spaced grid to equally-spaced grid.

    Parameters
    ----------
    f : (N, ) complex64
        Values of unequally-spaced function on the grid x
    x : (N, 3) float
        The frequencies on the unequally-spaced grid in the range [-0.5, 0.5)
    n : int
        The size of the equally-spaced grid along each edge.
    eps : float
        The accuracy of computing USFFT.
    scatter : function
        The scatter function to use.
    upsample : float >= 1
        The ratio of the upsampled grid to the equally-spaced grid.

    Returns
    -------
    F : (n, n, n) complex64
        The function at equally spaced frequencies. Frequencies on the grid are
        zero-centered i.e. [ -0.5, 0.25, 0.0,  0.25]
    """
    fftn = xp.fft.fftn if fftn is None else fftn
    upsampled = 2 * int(upsample * n / 2)  # upsampled grid is always even-sized
    pad = (upsampled - n) // 2  # where zero-padding stops
    end = pad + n  # where f stops

    # parameters for the USFFT transform
    mu = -xp.log(eps) / (2 * n**2)
    Te = 1 / xp.pi * xp.sqrt(-mu * xp.log(eps) + (mu * n)**2 / 4)
    m = int(xp.ceil(upsampled * Te))

    # smearing kernel (ker)
    kernel = _get_kernel(xp, n, mu)
    kernel *= upsampled**3

    G = scatter(xp, f, x, upsampled, m, mu)

    # FFT and compesantion for smearing
    F = checkerboard(xp, fftn(checkerboard(xp, G)), inverse=True)
    F = F[pad:end, pad:end, pad:end] / kernel

    return F


def _g(x):
    """Return -1 for odd x and 1 for even x."""
    return 1 - 2 * (x % 2)


def checkerboard(xp, array, axes=None, inverse=False):
    """In-place FFTshift for even sized grids only.

    If and only if the dimensions of `array` are even numbers, flipping the
    signs of input signal in an alternating pattern before an FFT is equivalent
    to shifting the zero-frequency component to the center of the spectrum
    before the FFT.
    """
    axes = range(array.ndim) if axes is None else axes
    for i in axes:
        if array.shape[i] % 2 != 0:
            raise ValueError(
                "Can only use checkerboard algorithm for even dimensions. "
                f"This dimension is {array.shape[i]}.")
        array = xp.moveaxis(array, i, -1)
        array *= _g(xp.arange(array.shape[-1]) + 1)
        if inverse:
            array *= _g(array.shape[-1] // 2)
        array = xp.moveaxis(array, -1, i)
    return array
