import numpy as np


def random_complex(*args):
    """Return a complex random array in the range (-0.5, 0.5)."""
    return (np.random.rand(*args) - 0.5 + 1j * (np.random.rand(*args) - 0.5))


def inner_complex(x, y):
    """Return the complex inner product; the order of the operands matters."""
    return np.sum(np.conj(x) * y)
