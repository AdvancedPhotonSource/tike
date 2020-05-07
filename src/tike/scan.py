#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2018, UChicago Argonne, LLC. All rights reserved.    #
#                                                                         #
# Copyright 2018. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################
"""Define building blocks of scanning trajectories and related functions.

Each trajectory returns position as a function of time and some other
parameters.

.. |t_docstring| replace:: Time steps to evaluate the function.
"""

__author__ = "Doga Gursoy, Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    'scantimes',
    'sinusoid',
    'triangle',
    'triangle_fs',
    'sawtooth',
    'square',
    'staircase',
    'lissajous',
    'raster',
    'spiral',
    'diagonal',
    'scan3',
    'avgspeed',
    'lengths',
    'distance',
    'billiard',
    'hexagonal',
]

import logging
import numpy as np

logger = logging.getLogger(__name__)


def _periodic_function_interface(t, A=0.5, f=60, p=0):
    """Define an exemplar periodic function for this module.

    Each trajectory function is a function of t, an array of time
    steps, and optional keyword arguements which determine the shape of
    the function. The function returns at least one spatial coordinate.

    Parameters
    ----------
    t : np.array
        |t_docstring|
    A : float
        The amplitude of the function.
    f : float
        The temporal frequency of the function.
    p : float [radians]
        The phase shift of the function.

    """
    raise NotImplementedError()


def hexagonal(t, D, f, row):
    """Hexagonal gridded step scan of circles with diameter, D.

    #discontinuous #2d

    Parameters
    ----------
    t : np.array
        |t_docstring|
    D : float
        The diameter of the circles.
    f : float
        How often to we move to the next circle.
    row : int
        The number of positions in each row

    """
    h = 0.5 * np.sqrt(3) * D
    x1 = staircase(A=h, f=f / row, p=0, t=t)
    x2 = (np.mod(staircase(A=D, f=f, p=0, t=t), row * D) + square(
        A=D * 0.25, f=f / row * 0.5, p=np.pi, t=t) + D * 0.25)
    return x1, x2


def f2w(f):
    """Return the angular frequency [rad] from the given frequency."""
    return 2 * np.pi * f


def period(f):
    """Return the period from the given frequency."""
    return 1 / f


def scantimes(t0, t1, f=60):
    """Return times in the range [t0, t1) at the given frequency (f)."""
    return np.linspace(t0, t1, (t1 - t0) * f, endpoint=False)


def sinusoid(A, f, p, t):
    """Return the value of a sine function at time `t`.

    #continuous #1D

    Parameters
    ----------
    t : np.array
        |t_docstring|
    A : float
        The amplitude of the function.
    f : float
        The temporal frequency of the function.
    p : float [radians]
        The phase shift of the function.

    """
    w = f2w(f)
    return A * np.sin(w * t - p)


def triangle(A, f, p, t):
    """Return the value of a triangle function at time `t`.

    #continuous #1d

    Parameters
    ----------
    t : np.array
        |t_docstring|
    A : float
        The amplitude of the function.
    f : float
        The temporal frequency of the function.
    p : float [radians]
        The phase shift of the function.

    """
    w = f2w(f)
    return A * 2 / np.pi * np.arcsin(np.sin(w * t - p))


def triangle_fs(A, f, p, t, N=8):
    """Approximate the triangle function using a Fourier series of N sinusoids.

    #continuous #1d
    """
    w = f2w(f)
    x = np.sin(w * t - p)
    for n in range(3, 2 * N, 2):
        x += (-1)**((n - 1) / 2) / (n * n) * np.sin(n * (w * t - p))

    return A * 8 / np.pi / np.pi * x


def sawtooth(A, f, p, t):
    """Return the value of a sawtooth function at time `t`.

    #discontinuous #1d

    Parameters
    ----------
    t : np.array
        |t_docstring|
    A : float
        The amplitude of the function.
    f : float
        The temporal frequency of the function.
    p : float [radians]
        The phase shift of the function.

    """
    ts = t * f - p / (2 * np.pi)
    q = np.floor(ts + 0.5)
    return A * (2 * (ts - q))


def square(A, f, p, t):
    """Return the value of a square function at time `t`.

    #discontinuous #1d

    Parameters
    ----------
    t : np.array
        |t_docstring|
    A : float
        The amplitude of the function.
    f : float
        The temporal frequency of the function.
    p : float [radians]
        The phase shift of the function.

    """
    ts = t - p / (2 * np.pi) / f
    return A * (np.power(-1, np.floor(2 * f * ts)))


def staircase(A, f, p, t):
    """Return the value of a staircase function at time `t`.

    #discontinuous #1d

    Parameters
    ----------
    t : np.array
        |t_docstring|
    A : float
        The amplitude of the function.
    f : float
        The temporal frequency of the function.
    p : float [radians]
        The phase shift of the function.

    """
    ts = t * f - p / (2 * np.pi)
    return A * np.floor(ts)


def lissajous(A, B, fx, fy, px, py, t):
    """Return the value of a lissajous function at time `t`.

    #continuous #2d
    The lissajous is centered on the origin. The lissajous is periodic if and
    only if the fx / fy is rational. The overall period is the
    least common multiple of the two periods.

    Parameters
    ----------
    t : np.array
        |t_docstring|
    A, B : float
        The horizontal and vertical amplitudes of the function
    fx, fy : float
        The temporal frequencies of the function
    px, py : float
        The phase shifts of the x and y components of the function

    """
    x = sinusoid(A, fx, px, t)
    y = sinusoid(B, fy, py, t)
    return x, y


def billiard(Ax, Ay, fx, fy, px, py, t, N):
    """Return the trajectory of a frictionless ball in a rectangular table.

    #continuous #2d
    It's a lissajous using triangle functions.
    """
    x = triangle_fs(Ax, fx, px, t, N)
    y = triangle_fs(Ay, fy, py, t, N)
    return x, y


def raster(A, B, f, x0, y0, t):
    """Return the value of a raster function at time `t`.

    #discontinuous #2d
    The raster starts at (x0, y0) and moves initially in the positive
    directions.

    Parameters
    ----------
    t : np.array
        |t_docstring|
    A : float
        The horizontal length of lines
    B : float
        The vertical space between lines
    f : float
        The number of raster lines per second
    x0, y0 : float
        Starting positions of the raster

    """
    x = 0.5 * (triangle(A, 0.5 * f, 0.5 * np.pi, t) + A) + x0
    y = staircase(B, f, 0, t) + y0
    return x, y


def spiral(r1, t1, v, t):
    """Return a spiral of constant linear velcity at time `t`.

    #continuous #2d
    The spiral is centered on the origin and spins clockwise.

    Parameters
    ----------
    t : np.array
        |t_docstring|
    r1 : float
        The radius at time t1.
    t1: float
        The time at which the spiral reaches r1.
    v : float
        The linear velocity of the trajectory

    References
    ----------
    A. Bazaei, M. Maroufi and S. O. R. Moheimani, "Tracking of
    constant-linear-velocity spiral trajectories by approximate internal model
    control," 2017 IEEE Conference on Control Technology and Applications
    (CCTA), Mauna Lani, HI, 2017, pp. 129-134. doi: 10.1109/CCTA.2017.8062452

    """
    P = np.pi * r1 * r1 / t1 / v
    r = np.sqrt(P * v * t / np.pi)
    theta = 2 * np.sqrt(np.pi * v * t / P)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def diagonal(A, B, fx, fy, px, py, t):
    """Return the value of a diagonal function at time `t`.

    #discontinuous #2d
    The diagonal is centered on the origin.

    Parameters
    ----------
    A, B : float
        The horizontal and vertical amplitudes of the function
    fx, fy : float
        The temporal frequencies of the function
    px, py : float
        The phase shifts of the x and y components of the function
    t : np.array
        |t_docstring|

    """
    x = triangle(A, fx, px + np.pi / 2, t)
    y = triangle(B, fy, py + np.pi / 2, t)
    return x, y


def scan3(A, B, fx, fy, fz, px, py, time, hz):
    """Return a 3D combination of lissajous and sawtooth trajectories."""
    x, y, t = lissajous(A, B, fx, fy, px, py, time, hz)
    z = sawtooth(np.pi, 0.5 * fz, 0.5 * np.pi, t, hz)
    return z, x, y, t


def avgspeed(time, x, y=None, z=None):
    """Return the average speed along trajectory x, y, z if covered in time."""
    return distance(z, x, y) / time


def lengths(x, y=None, z=None):
    """Return the absolute displacements between points defined by x, y, z."""
    if y is None:
        y = np.zeros(x.shape)
    if z is None:
        z = np.zeros(x.shape)
    a = np.diff(x)
    b = np.diff(y)
    c = np.diff(z)
    return np.sqrt(a * a + b * b + c * c)


def distance(x, y=None, z=None):
    """Return the total distance travelled along the trajectory x, y, z."""
    d = lengths(z, x, y)
    return np.sum(d)
