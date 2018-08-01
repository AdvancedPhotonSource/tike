#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2017-2018, UChicago Argonne, LLC. All rights reserved.    #
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

"""
Module for scanning building blocks
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import logging
from math import sqrt, atan2, cos

__author__ = "Doga Gursoy, Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['scantimes',
           'sinusoid',
           'triangle',
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
           'distance']


logger = logging.getLogger(__name__)


def f2w(f):
    """Return the angular frequency from the given frequency"""
    return 2*np.pi*f


def period(f):
    """Return the period from the given frequency"""
    return 1 / f


def scantimes(t0, t1, hz):
    """An array of points in the range [t0, t1) at the given frequency (hz)
    """
    return np.linspace(t0, t1, (t1-t0)*hz, endpoint=False)


def sinusoid(A, f, p, t):
    """Return the value of a sine function at time `t`.
    #continuous #1D

    Parameters
    A : float
        The amplitude of the function
    f : float
        The temporal frequency of the function
    p : float
        The phase shift of the function
    """
    w = f2w(f)
    return A * np.sin(w*t - p)


def triangle(A, f, p, t):
    """Return the value of a triangle function at time `t`.
    #continuous #1d

    Parameters
    A : float
        The amplitude of the function
    f : float
        The temporal frequency of the function
    p : float
        The phase shift of the function
    """
    a = 0.5 / f
    ts = t - p/(2*np.pi)/f
    q = np.floor(ts/a + 0.5)
    return A * (2/a * (ts - a*q) * np.power(-1, q))


def sawtooth(A, f, p, t):
    """Return the value of a sawtooth function at time `t`.
    #discontinuous #1d

    Parameters
    A : float
        The amplitude of the function
    f : float
        The temporal frequency of the function
    p : float
        The phase shift of the function
    """
    ts = t*f - p/(2*np.pi)
    q = np.floor(ts + 0.5)
    return A * (2 * (ts - q))


def square(A, f, p, t):
    """Return the value of a square function at time `t`.
    #discontinuous #1d

    Parameters
    A : float
        The amplitude of the function
    f : float
        The temporal frequency of the function
    p : float
        The phase shift of the function
    """
    ts = t - p/(2*np.pi)/f
    return A * (np.power(-1, np.floor(2*f*ts)))


def staircase(A, f, p, t):
    """Return the value of a staircase function at time `t`.
    #discontinuous #1d

    Parameters
    A : float
        The amplitude of the function
    f : float
        The temporal frequency of the function
    p : float
        The phase shift of the function
    """
    ts = t*f - p/(2*np.pi)
    return A * np.floor(ts)


def lissajous(A, B, fx, fy, px, py, t):
    """Return the value of a lissajous function at time `t`.
    #continuous #2d

    The lissajous is centered on the origin. The lissajous is periodic if and
    only if the fx / fy is rational. The overall period is the
    least common multiple of the two periods.

    Parameters
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


def raster(A, B, f, x0, y0, t):
    """Return the value of a raster function at time `t`.
    #discontinuous #2d

    The raster starts at (x0, y0) and moves initially in the positive
    directions.

    Parameters
    ----------
    A : float
        The horizontal length of lines
    B : float
        The vertical space between lines
    f : float
        The number of raster lines per second
    x0, y0 : float
        Starting positions of the raster
    """
    x = 0.5 * (triangle(A, 0.5*f, 0.5*np.pi, t) + A) + x0
    y = staircase(B, f, 0, t) + y0
    return x, y


def spiral(r1, t1, v, t):
    """Return a spiral of constant linear velcity at time `t`.
    #continuous #2d

    The spiral is centered on the origin and spins clockwise.

    Parameters
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
    A, B : float
        The horizontal and vertical amplitudes of the function
    fx, fy : float
        The temporal frequencies of the function
    px, py : float
        The phase shifts of the x and y components of the function
    """
    x = triangle(A, fx, px+np.pi/2, t)
    y = triangle(B, fy, py+np.pi/2, t)
    return x, y


def scan3(A, B, fx, fy, fz, px, py, time, hz):
    x, y, t = lissajous(A, B, fx, fy, px, py, time, hz)
    z = sawtooth(np.pi, 0.5*fz, 0.5*np.pi, t, hz)
    return z, x, y, t


def avgspeed(time, x, y=None, z=None):
    return distance(z, x, y) / time


def lengths(x, y=None, z=None):
    if y is None:
        y = np.zeros(x.shape)
    if z is None:
        z = np.zeros(x.shape)
    a = np.diff(x)
    b = np.diff(y)
    c = np.diff(z)
    return sqrt(a*a + b*b + c*c)


def distance(x, y=None, z=None):
    d = lengths(z, x, y)
    return np.sum(d)
