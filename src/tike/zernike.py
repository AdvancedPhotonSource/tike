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
    if m < 0:
        polynomial = R(_m_, n, radius) * np.sin(m * angle)
    else:
        polynomial = R(_m_, n, radius) * np.cos(m * angle)
    polynomial[np.logical_or(radius < 0, radius > 1)] = 0
    return polynomial


def N(m: int, n: int) -> float:
    """Zernike normalization factor."""
    return np.sqrt(2 * (n + 1) / (1 + (m == 0)))


def R(m: int, n: int, radius: np.array) -> np.array:
    """Zernike radial polynomial."""
    # Initialize with k=0 case because this term will always be included
    sign = -1
    result = 0 * radius
    for k in range(0, (n - m) // 2 + 1):
        sign = -sign
        result += sign * bino(n, m, k) * radius**(n - 2 * k)
    return result


def bino(n: int, m: int, k: int) -> int:
    """Return the approximate binomial coeffient (a b)."""
    return int(
        factorial(n - k) / factorial(k) / factorial((n + m) // 2 - k) /
        factorial((n - m) // 2 - k))


def zernike_basis(size: int, degree: int) -> np.array:
    """Return all circular Zernike basis for radial degree up to n."""
    endpoint = 1.0 - 1.0 / (2 * size)
    x = np.linspace(-endpoint, endpoint, size, endpoint=True)
    coords = np.meshgrid(x, x, indexing='ij')
    radius = np.linalg.norm(coords, axis=0)
    theta = np.arctan2(coords[0], coords[1])

    basis = []
    for m, n in valid_zernike_indices(degree):
        basis.append(Z(m, n, radius, theta))

    basis = np.stack(basis, axis=0)
    return basis


def valid_zernike_indices(degree):
    for n in range(0, degree):
        for m in range(-n, n + 1):
            if (n - abs(m)) % 2 == 0:
                yield m, n
