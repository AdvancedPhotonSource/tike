
def sequential_gather(xp, f, x, n, m, mu, ndim=3):
    w = n + m
    cons = [xp.sqrt(xp.pi / mu)**3, -xp.pi**2 / mu]
    G = xp.zeros([2 * (n + m)] * ndim, dtype="complex64")
    for i0 in range(-w, w):
        for i1 in range(-w, w):
            for i2 in range(-w, w):
                for k in range(x.shape[0]):
                    if (abs(i0 - xp.floor(2 * n * x[k, 0])) < m
                            and abs(i1 - xp.floor(2 * n * x[k, 1])) < m
                            and abs(i2 - xp.floor(2 * n * x[k, 2])) < m):
                        Fkernel = cons[0] * xp.exp(cons[1] * (
                            + (i0 / (2 * n) - x[k, 0])**2
                            + (i1 / (2 * n) - x[k, 1])**2
                            + (i2 / (2 * n) - x[k, 2])**2
                        ))  # yapf: disable
                        G[i0 + w, i1 + w, i2 + w] += f[k] * Fkernel
    return G



def vector_gather(xp, f, x, n, m, mu, ndim=3):
    """
    Parameters
    ----------
    f : (N, )
        values at frequencies
    x : (N, 3)
        non-uniform frequencies

    Return
    ------
    G : [2 *(n + m)] * 3 array

    """
    w = n + m
    cons = [xp.sqrt(xp.pi / mu)**3, -xp.pi**2 / mu]
    G = xp.zeros([2 * (n + m)] * ndim, dtype="complex64")
    mf = (m / (2 * n))**2  # kernel radius
    for i0 in range(-w, w):
        for i1 in range(-w, w):
            for i2 in range(-w, w):
                delta0 = (i0 / (2 * n) - x[:, 0])**2
                delta1 = (i1 / (2 * n) - x[:, 1])**2
                delta2 = (i2 / (2 * n) - x[:, 2])**2
                k = xp.logical_and.reduce((
                    delta0 < mf,
                    delta1 < mf,
                    delta2 < mf,
                ))
                Fkernel = cons[0] * xp.exp(cons[1] * (
                    + delta0[k]
                    + delta1[k]
                    + delta2[k]
                ))  # yapf: disable
                G[i0 + w, i1 + w, i2 + w] = xp.sum(f[k] * Fkernel)
    return G
