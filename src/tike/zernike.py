import numpy as np


def Z(m, n, ρ, θ):
    assert np.all(0 <= ρ <= 1), "Radii must be in range [0, 1]."
    assert n >= 0, "Radial degree must be non-negative."
    _m_ = np.abs(m)
    assert _m_ <= n, "Angular frequency must be less than radial degree."
    return N(m, n) * R(_m_, n, ρ) * np.exp(1j * _m_ * θ)


def N(m, n):
    """Zernike normalization factor."""
    # @StevenHenke1 this must be floating point division?
    return np.sqrt(2 * (n + 1) / (1 + (m == 0)))


def R(m, n, ρ):
    """Zernike radial polynomial."""
    if (n - m) % 2:
        return 0
    else:
        # Initialize with k=0 case because this term will always be included
        sign = 1
        b0 = 1
        b1 = 1
        result = ρ**n
        # @StevenHenke2 these must be integer division?
        # @StevenHenke3 Does this sum include k = (n - m) // 2 ?
        for k in range(1, (n - m) // 2 + 1):
            sign = -sign
            b0 *= bino(n - k, k)
            b1 *= bino(n - 2 * k, (n - m) // 2 - k)
            result += sign * b0 * b1 * ρ**(n - 2 * k)
        return result


def bino(n, i):
    """One product term of the binomial coefficient."""
    # @StevenHenke4 these must be integer division?
    assert i >= 0
    if i == 0:
        return 1
    else:
        return (n - i + 1) // i
