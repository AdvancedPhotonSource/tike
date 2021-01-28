import cupy as cp
import numpy as np


def add_modes_random_phase(probe, nmodes):
    """Initialize additional probe modes by phase shifting the first mode.

    Parameters
    ----------
    probe : (:, :, :, M, :, :) array
        A probe with M > 0 incoherent modes.
    nmodes : int
        The number of desired modes.

    References
    ----------
    M. Odstrcil, P. Baksh, S. A. Boden, R. Card, J. E. Chad, J. G. Frey, W. S.
    Brocklesby, "Ptychographic coherent diffractive imaging with orthogonal
    probe relaxation." Opt. Express 24, 8360 (2016). doi: 10.1364/OE.24.008360
    """
    all_modes = np.empty((*probe.shape[:-3], nmodes, *probe.shape[-2:]),
                         dtype='complex64')
    pw = probe.shape[-1]
    for m in range(nmodes):
        if m < probe.shape[-3]:
            # copy existing mode
            all_modes[..., m, :, :] = probe[..., m, :, :]
        else:
            # randomly shift the first mode
            shift = np.exp(-2j * np.pi * (np.random.rand(2, 1) - 0.5) *
                           ((np.arange(0, pw) + 0.5) / pw - 0.5))
            all_modes[..., m, :, :] = (probe[..., 0, :, :] * shift[0][None] *
                                       shift[1][:, None])
    return all_modes


def orthogonalize_eig(x):
    """Orthogonalize modes of x using eigenvectors of the pairwise dot product.

    Parameters
    ----------
    x : (..., nmodes, :, :) array_like complex64
        An array of the probe modes vectorized

    References
    ----------
    M. Odstrcil, P. Baksh, S. A. Boden, R. Card, J. E. Chad, J. G. Frey, W. S.
    Brocklesby, "Ptychographic coherent diffractive imaging with orthogonal
    probe relaxation." Opt. Express 24, 8360 (2016). doi: 10.1364/OE.24.008360
    """
    nmodes = x.shape[-3]
    # 'A' holds the dot product of all possible mode pairs. We only fill the
    # lower half of `A` because it is conjugate-symmetric
    A = cp.empty((*x.shape[:-3], nmodes, nmodes), dtype='complex64')
    for i in range(nmodes):
        for j in range(i + 1):
            A[..., i, j] = cp.sum(cp.conj(x[..., i, :, :]) * x[..., j, :, :],
                                  axis=(-1, -2))

    _, vectors = cp.linalg.eigh(A, UPLO='L')
    # np.linalg.eigh guarantees that the eigen values are returned in ascending
    # order, so we just reverse the order of modes to have them sorted in
    # descending order.

    # TODO: Optimize this double-loop
    x_new = cp.zeros_like(x)
    for i in range(nmodes):
        for j in range(nmodes):
            # Sort new modes by eigen value in decending order.
            x_new[..., nmodes - 1 -
                  j, :, :] += vectors[..., i, j, None, None] * x[..., i, :, :]
    assert x_new.shape == x.shape, [x_new.shape, x.shape]

    return x_new


if __name__ == "__main__":
    cp.random.seed(0)
    x = (cp.random.rand(7, 1, 9, 3, 3) +
         1j * cp.random.rand(7, 1, 9, 3, 3)).astype('complex64')
    x1 = orthogonalize_eig(x)
    assert x1.shape == x.shape, x1.shape
