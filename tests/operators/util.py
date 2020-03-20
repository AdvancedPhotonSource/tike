
import numpy as np

def random_complex(*args):
    """Return a complex random array in the range (-0.5, 0.5)."""
    return (        np.random.rand(*args) - 0.5
            + 1j * (np.random.rand(*args) - 0.5))
