import numpy as np

def run(xp, u, mu, tau, alpha):
    """Provide some kind of regularization."""
    z = fwd(xp, u) + mu / tau
    # Soft-thresholding
    # za = xp.sqrt(xp.sum(xp.abs(z), axis=0))
    za = xp.sqrt(xp.real(xp.sum(z*xp.conj(z), 0)))
    zeros = (za <= alpha / tau)
    z[:, zeros] = 0
    z[:, ~zeros] -= z[:, ~zeros] * alpha / (tau * za[~zeros])
    return z

def fwd(xp, u):
    """Forward operator for regularization (J)."""
    res = xp.zeros((3, *u.shape), dtype=u.dtype, order='C')
    res[0, :, :, :-1] = u[:, :, 1:] - u[:, :, :-1]
    res[1, :, :-1, :] = u[:, 1:, :] - u[:, :-1, :]
    res[2, :-1, :, :] = u[1:, :, :] - u[:-1, :, :]
    res *= 2 / np.sqrt(3)  # normalization
    return res

def adj(xp, gr):
    """Adjoint operator for regularization (J^*)."""
    res = xp.zeros(gr.shape[1:], gr.dtype, order='C')
    res[:, :, 1:] = gr[0, :, :, 1:] - gr[0, :, :, :-1]
    res[:, :, 0] = gr[0, :, :, 0]
    res[:, 1:, :] += gr[1, :, 1:, :] - gr[1, :, :-1, :]
    res[:, 0, :] += gr[1, :, 0, :]
    res[1:, :, :] += gr[2, 1:, :, :] - gr[2, :-1, :, :]
    res[0, :, :] += gr[2, 0, :, :]
    res *= -2 / np.sqrt(3)  # normalization
    return res
