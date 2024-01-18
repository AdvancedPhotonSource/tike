"""Provide functions to evaluate Zernike polynomials on a discrete grid."""

import numpy as np


def Z(m: int, n: int, radius: np.array, angle: np.array) -> np.array:
    """Return values of Zernike[m,n] polynomial at given radii, angles.

    Values outside valid radius will be zero.

    Parameters
    ----------
    m : int
        Angular frequency of the polynomial.
    n : int
        Radial degree of the polynomial.
    radius: float [0, 1]
        The radial coordinates of the evaluated polynomial.
    angle: float radians
        The angular coordinates of the evaluated polynomial.

    """
    if n < 0:
        raise ValueError("Radial degree must be non-negative.")
    _m_ = np.abs(m)
    if _m_ > n:
        raise ValueError("Angular frequency must be less than radial degree.")
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
    """Return the values of the Zernike radial polynomial at the given radii.

    This polynomial matches Figure 3 in Lakshminarayanan & Fleck (2011).

    Parameters
    ----------
    m : int
        Angular frequency of the polynomial.
    n : int
        Radial degree of the polynomial.
    radius: float [0, 1]
        The radial coordinates of the evaluated polynomial.

    References
    ----------
    Vasudevan Lakshminarayanan & Andre Fleck (2011): Zernike polynomials: a
    guide, Journal of ModernOptics, 58:7, 545-561
    http://dx.doi.org/10.1080/09500340.2011.554896

    """
    # Initialize with k=0 case because this term will always be included
    sign = -1
    result = 0 * radius
    for k in range(0, (n - m) // 2 + 1):
        sign = -sign
        b0 = _bino(n - k, k)
        b1 = _bino(n - 2 * k, (n - m) // 2 - k)
        result += sign * b0 * b1 * radius ** (n - 2 * k)
    return result


def _bino(a: int, b: int) -> int:
    """Return the approximate binomial coeffient (a b)."""
    result = 1
    for i in range(1, b + 1):
        result *= (a - i + 1) / i
    return result


def _bino1(a: int, b: int, xp=np) -> int:
    """Return the approximate binomial coeffient (a b)."""
    result = np.arange(a, a - b, -1) / np.arange(1, b + 1)
    return np.prod(result)


def basis(size: int, degree_min: int, degree_max: int, xp=np) -> np.array:
    """Return all circular Zernike basis up to given radial degree.

    Parameters
    ----------
    size : int
        The width of the discrete basis in pixel.
    degree : int
        The maximum radial degree of the polynomial (not inclusive). The number
        of degrees included in the set of bases.

    Returns
    -------
    basis : (degree, size, size)
        The Zernike bases.

    """
    endpoint = 1.0 - 1.0 / (2 * size)
    x = xp.linspace(-endpoint, endpoint, size, endpoint=True)
    coords = xp.stack(xp.meshgrid(x, x, indexing="ij"), axis=0)
    radius = xp.linalg.norm(coords, axis=0)
    theta = xp.arctan2(coords[0], coords[1])

    basis = []
    for m, n in valid_indices(degree_min, degree_max):
        basis.append(Z(m, n, radius, theta))

    basis = xp.stack(basis, axis=0)
    return basis


def valid_indices(degree_min: int, degree_max: int) -> tuple:
    """Enumerate all valid zernike indices (m,n) up to the given degree."""
    for n in range(degree_min, degree_max):
        for m in range(-n, n + 1):
            if (n - abs(m)) % 2 == 0:
                yield m, n


def degree_from_num_coeffients(num_coefficients: int) -> int:
    coefficient_count = 0
    for n in range(0, 999_999):
        for m in range(-n, n + 1):
            if (n - abs(m)) % 2 == 0:
                coefficient_count += 1
        if coefficient_count >= num_coefficients:
            return n + 1, coefficient_count
