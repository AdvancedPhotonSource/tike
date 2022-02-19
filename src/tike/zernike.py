"""Provide functions to generate complex coefficients of Zernike polynomials
on a discrete grid."""

from math import factorial
import numpy as np


def Z(m: int, n: int, radius: np.array, angle: np.array) -> np.array:
    """Return the coefficients of the Zernike[m,n] polynomial.

    Parameters
    ----------
    m : int
        angular frequency
    n : int
        radial degree
    radius: float [0, 1]
        radius
    angle: float radians
        angle
    """
    assert n >= 0, "Radial degree must be non-negative."
    _m_ = np.abs(m)
    assert _m_ <= n, "Angular frequency must be less than radial degree."
    polynomial = R(_m_, n, radius) * np.exp(1j * m * angle)
    polynomial[np.logical_or(radius < 0, radius > 1)] = np.nan
    return polynomial


def N(m: int, n: int) -> int:
    """Zernike normalization factor."""
    return np.sqrt(2 * (n + 1) / (1 + (m == 0)))


def R(m: int, n: int, radius: np.array) -> np.array:
    """Zernike radial polynomial."""
    # Initialize with k=0 case because this term will always be included
    sign = 1
    result = bino(n, m, 0) * radius**n
    for k in range(1, n - m + 1):
        sign = -sign
        result += sign * bino(n, m, k) * radius**(n - k)
    return result


def bino(n: int, m: int, k: int) -> int:
    """Return the approximate binomial coeffient (a b)."""
    return int(
        factorial(2 * n + 1 - k) / factorial(k) / factorial(n - m - k) /
        factorial(n + m - k + 1))


def mode(size: int, n: int) -> np.array:
    endpoint = 1.0 - 1 / (2 * size)
    x = np.linspace(-endpoint, endpoint, size, endpoint=True)
    coords = np.meshgrid(x, x, indexing='ij')
    radius = np.linalg.norm(coords, axis=0)
    theta = np.arctan(coords[0] / coords[1])

    basis = []
    for _n in range(0, n):
        for m in range(-_n, _n + 1):
            basis.append(Z(m, n, radius, theta))

    basis = np.stack(basis, axis=0)
    return basis
