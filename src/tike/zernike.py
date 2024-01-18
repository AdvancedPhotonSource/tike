"""Provide functions to evaluate Zernike polynomials on a discrete grid.


References
----------
@article{Niu_2022,
doi = {10.1088/2040-8986/ac9e08},
url = {https://dx.doi.org/10.1088/2040-8986/ac9e08},
year = {2022},
month = {nov},
publisher = {IOP Publishing},
volume = {24},
number = {12},
pages = {123001},
author = {Kuo Niu and Chao Tian},
title = {Zernike polynomials and their applications},
journal = {Journal of Optics},
abstract = {The Zernike polynomials are a complete set of continuous functions orthogonal over a unit circle. Since first developed by Zernike in 1934, they have been in widespread use in many fields ranging from optics, vision sciences, to image processing. However, due to the lack of a unified definition, many confusing indices have been used in the past decades and mathematical properties are scattered in the literature. This review provides a comprehensive account of Zernike circle polynomials and their noncircular derivatives, including history, definitions, mathematical properties, roles in wavefront fitting, relationships with optical aberrations, and connections with other polynomials. We also survey state-of-the-art applications of Zernike polynomials in a range of fields, including the diffraction theory of aberrations, optical design, optical testing, ophthalmic optics, adaptive optics, and image analysis. Owing to their elegant and rigorous mathematical properties, the range of scientific and industrial applications of Zernike polynomials is likely to expand. This review is expected to clear up the confusion of different indices, provide a self-contained reference guide for beginners as well as specialists, and facilitate further developments and applications of the Zernike polynomials.}
}
"""

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
        return np.sqrt(2 * (n + 1)) * R(_m_, n, radius) * np.sin(m * angle)
    if m == 0:
        return np.sqrt(n + 1) * R(_m_, n, radius)
    if m > 0:
        return np.sqrt(2 * (n + 1)) * R(_m_, n, radius) * np.cos(m * angle)


def N(m: int, n: int) -> float:
    """Zernike normalization factor."""
    if m == 0:
        return np.sqrt(n + 1)
    return np.sqrt(2 * (n + 1))


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
    result[radius > 1] = 0.0
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
    """
    There are a total of (n + 1)(n + 2)/2 linearly independent polynomials for a degree ⩽ n.
    """
    coefficient_count = 0
    for n in range(0, 999_999):
        for m in range(-n, n + 1):
            if (n - abs(m)) % 2 == 0:
                coefficient_count += 1
        if coefficient_count >= num_coefficients:
            return n + 1, coefficient_count
