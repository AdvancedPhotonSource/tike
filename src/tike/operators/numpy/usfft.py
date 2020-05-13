def _get_kernel(xp, pad, mu):
    """Return the interpolation kernel for the USFFT."""
    xeq = xp.mgrid[-pad:pad, -pad:pad, -pad:pad]
    return xp.exp(-mu * xp.sum(xeq**2, axis=0)).astype('float32')


def eq2us(f, x, n, eps, xp):
    """USFFT from equally-spaced grid to unequally-spaced grid.

    Parameters
    ----------
    f : (n, n, n) complex64
        The function to be transformed on a regular-grid of size n.
    x : (, 3)
        The sampled frequencies on unequally-spaced grid.
    eps : float
        The desired relative?/absolute? accuracy of the USFFT.
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

    # wrapping array Fe0
    idx = xp.mgrid[-m:2 * n + m, -m:2 * n + m, -m:2 * n + m]
    idx0 = xp.mod(idx + 2 * n, 2 * n)
    Fe = xp.zeros([2 * (n + m)] * ndim, dtype="complex64")
    Fe[tuple(idx + m)] = Fe0[tuple(idx0)]

    # smearing operation (F=Fe*kera), gathering
    cons = [xp.sqrt(xp.pi / mu)**3, xp.pi**2 / mu]

    # # Sequential approach (slow)
    # F = xp.zeros(x.shape[0], dtype="complex64")
    # for k in range(x.shape[0]):
    #     ell0 = xp.int(xp.floor(2 * n * x[k, 0]))
    #     ell1 = xp.int(xp.floor(2 * n * x[k, 1]))
    #     ell2 = xp.int(xp.floor(2 * n * x[k, 2]))
    #     for i0 in range(2 * m):
    #         for i1 in range(2 * m):
    #             for i2 in range(2 * m):
    #                 kera = cons[0] * xp.exp(cons[1] * (
    #                     + ((ell0 - m + i0) / (2 * n) - x[k, 0])**2
    #                     + ((ell1 - m + i1) / (2 * n) - x[k, 1])**2
    #                     + ((ell2 - m + i2) / (2 * n) - x[k, 2])**2
    #                 ))  # yapf: disable
    #                 F[k] += Fe[n + ell0 + i0, n + ell1 + i1, n + ell2 + i2] * kera

    # # Vectorize kernel (fast for large kernels)
    # i = xp.mgrid[0:2 * m, 0:2 * m, 0:2 * m]
    # F = xp.empty(x.shape[0], dtype="complex64")
    # for k in range(x.shape[0]):
    #     ell = xp.floor(2 * n * x[k]).astype('int')[:, None, None, None]
    #     kera = cons[0] * xp.exp(cons[1] * xp.sum(
    #         xp.square((ell - m + i) / (2 * n) - x[k][:, None, None, None]),
    #         axis=0,
    #     ))
    #     F[k] = xp.sum(Fe[tuple(n + ell + i)] * kera)

    # Vectorize approach (faster)
    F = xp.zeros(x.shape[0], dtype="complex64")
    Fe = Fe.ravel()
    ell0 = ((2 * n * x[:, 0]) // 1).astype(xp.int32)
    ell1 = ((2 * n * x[:, 1]) // 1).astype(xp.int32)
    ell2 = ((2 * n * x[:, 2]) // 1).astype(xp.int32)

    for i0 in range(2 * m):
        delta0 = (ell0 - m + i0).astype('float32') / (2 * n) - x[:, 0]
        for i1 in range(2 * m):
            delta1 = (ell1 - m + i1).astype('float32') / (2 * n) - x[:, 1]
            for i2 in range(2 * m):
                delta2 = (ell2 - m + i2).astype('float32') / (2 * n) - x[:, 2]

                kera = cons[0] * xp.exp(-cons[1] *
                                        (delta0**2 + delta1**2 + delta2**2))
                ids = n+ell2+i2+(2*n+2*m)*(n+ell1+i1) + \
                    (2*n+2*m)*(2*n+2*m)*(n+ell0+i0)
                F += Fe[ids] * kera

    return F


def us2eq(f, x, n, eps, xp):
    """
        USFFT from unequally-spaced grid to equally-spaced grid
        x - unequally-spaced grid
        f - function on the grid x
        eps - accuracy of computing USFFT
    """
    # parameters for the USFFT transform
    mu = -xp.log(eps) / (2 * n**2)
    Te = 1 / xp.pi * xp.sqrt(-mu * xp.log(eps) + (mu * n)**2 / 4)
    m = xp.int(xp.ceil(2 * n * Te))

    # smearing kernel (ker)
    ker = xp.zeros((2 * n, 2 * n, 2 * n), dtype="complex64")
    [xeq0, xeq1, xeq2] = xp.mgrid[-n // 2:n // 2, -n // 2:n // 2,
                                  -n // 2:n // 2]
    ker[n // 2:n // 2 + n, n // 2:n // 2 + n,
        n // 2:n // 2 + n] = xp.exp(-mu * xeq0**2 - mu * xeq1**2 - mu * xeq2**2)

    # smearing operation (G=f*kera)
    G = xp.zeros([(2 * n + 2 * m) * (2 * n + 2 * m) * (2 * n + 2 * m)],
                 dtype="complex64")

    # Sequential approach (slow)
    # for k in range(x.shape[0]):
    #     ell0 = xp.int(xp.floor(2*n*x[k, 0]))
    #     ell1 = xp.int(xp.floor(2*n*x[k, 1]))
    #     ell2 = xp.int(xp.floor(2*n*x[k, 2]))
    #     for i0 in range(2*m):
    #         for i1 in range(2*m):
    #             for i2 in range(2*m):
    #                 kera = xp.sqrt(xp.pi)**3/xp.sqrt(mu*mu*mu)*(xp.exp( -xp.pi**2/mu*((ell0-m+i0)/(2*n)-x[k, 0])**2
    #                                                                     -xp.pi**2/mu*((ell1-m+i1)/(2*n)-x[k, 1])**2
    #                                                                     -xp.pi**2/mu*((ell2-m+i2)/(2*n)-x[k, 2])**2))
    #                 G[n+ell0+i0, n+ell1+i1, n+ell2+i2] += f[k]*kera

    # Vectorize approach (faster)
    ell0 = ((2 * n * x[:, 0]) // 1).astype(xp.int32)
    ell1 = ((2 * n * x[:, 1]) // 1).astype(xp.int32)
    ell2 = ((2 * n * x[:, 2]) // 1).astype(xp.int32)
    # precomute constants
    cons = [xp.sqrt(xp.pi / mu)**3, xp.pi**2 / mu]

    for i0 in range(2 * m):
        delta0 = (ell0 - m + i0).astype('float32') / (2 * n) - x[:, 0]
        for i1 in range(2 * m):
            delta1 = (ell1 - m + i1).astype('float32') / (2 * n) - x[:, 1]
            for i2 in range(2 * m):
                delta2 = (ell2 - m + i2).astype('float32') / (2 * n) - x[:, 2]

                kera = cons[0] * xp.exp(-cons[1] *
                                        (delta0**2 + delta1**2 + delta2**2))
                ids = n+ell2+i2+(2*n+2*m)*(n+ell1+i1) + \
                    (2*n+2*m)*(2*n+2*m)*(n+ell0+i0)
                vals = f * kera
                # accumulate by indexes (with possible index intersections),
                # TODO acceleration of bincount!!
                vals = xp.bincount(ids, weights=vals.real) + \
                    1j*xp.bincount(ids, weights=vals.imag)
                ids = xp.nonzero(vals)[0]
                G[ids] += vals[ids]

    # wrapping array G
    [idx, idy, idz] = xp.mgrid[-m:2 * n + m, -m:2 * n + m, -m:2 * n + m]
    idx0 = xp.mod(idx + 2 * n, 2 * n)
    idy0 = xp.mod(idy + 2 * n, 2 * n)
    idz0 = xp.mod(idz + 2 * n, 2 * n)
    ind = (idz0 + idy0 * (2 * n) + idx0 * (2 * n * 2 * n)).flatten()
    # accumulate by indexes (with possible index intersections)
    G = xp.bincount(ind, weights=G.real) + 1j * xp.bincount(ind, weights=G.imag)
    G = G.reshape([2 * n, 2 * n, 2 * n])

    # FFT and compesantion for smearing
    F = xp.fft.fftshift(xp.fft.fftn(xp.fft.fftshift(G)))
    F = F[n//2:n//2+n, n//2:n//2+n, n//2:n//2+n] / \
        ker[n//2:n//2+n, n//2:n//2+n, n//2:n//2+n]/(2*n*2*n*2*n)

    return F
