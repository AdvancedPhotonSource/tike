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


# TODO: Possibly a faster implementation would use QR decomposition, but numpy
# only support 2D inputs for QR as of 2020.04.
def orthogonalize_gs(x, ndim=1):
    """Gram-schmidt orthogonalization for complex arrays.

    x : (..., nmodes, :, :) array_like
        The array with modes in the -3 dimension.

    ndim : int > 0
        The number of trailing dimensions to orthogonalize.

    """
    if ndim < 1:
        raise ValueError("Must orthogonalize at least one dimension!")

    def inner(x, y, axis=None):
        """Return the complex inner product of x and y along axis."""
        return np.sum(np.conj(x) * y, axis=axis, keepdims=True)

    unflat_shape = x.shape
    nmodes = unflat_shape[-ndim - 1]
    x_ortho = x.reshape(*unflat_shape[:-ndim], -1)

    for i in range(1, nmodes):
        u = x_ortho[..., 0:i, :]
        v = x_ortho[..., i:i + 1, :]
        projections = u * inner(u, v, axis=-1) / inner(u, u, axis=-1)
        x_ortho[..., i:i + 1, :] -= np.sum(projections, axis=-2, keepdims=True)

    if __debug__:
        # Test each pair of vectors for orthogonality
        for i in range(nmodes):
            for j in range(i):
                error = abs(
                    inner(x_ortho[..., i:i + 1, :],
                          x_ortho[..., j:j + 1, :],
                          axis=-1))
                assert np.all(error < 1e-5), (
                    f"Some vectors are not orthogonal!, {error}, {error.shape}")

    return x_ortho.reshape(unflat_shape)


def orthogonalize_eig(x):
    """Orthogonalize modes of x using eigenvectors of the pairwise dot product.

    Parameters
    ----------
    x : (nmodes, probe_shape * probe_shape) array_like complex64
        An array of the probe modes vectorized

    References
    ----------
    M. Odstrcil, P. Baksh, S. A. Boden, R. Card, J. E. Chad, J. G. Frey, W. S.
    Brocklesby, "Ptychographic coherent diffractive imaging with orthogonal
    probe relaxation." Opt. Express 24, 8360 (2016). doi: 10.1364/OE.24.008360
    """
    nmodes = x.shape[0]
    # 'A' holds the dot product of all possible mode pairs
    A = np.empty((nmodes, nmodes), dtype='complex64')
    for i in range(nmodes):
        for j in range(nmodes):
            A[i, j] = np.sum(np.conj(x[i]) * x[j])

    values, vectors = np.linalg.eig(A)

    x_new = np.zeros_like(x)
    for i in range(nmodes):
        for j in range(nmodes):
            x_new[j] += vectors[i, j] * x[i]

    # Sort new modes by eigen value in decending order
    return x_new[np.argsort(-values)]
