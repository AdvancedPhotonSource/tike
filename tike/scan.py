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
Module for 3D scanning.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import logging


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['scantime',
           'sinusoid',
           'triangle',
           'sawtooth',
           'square',
           'staircase',
           'lissajous',
           'raster',
           'spiral',
           'scan3',
           'avgspeed',
           'lengths',
           'distance']


logger = logging.getLogger(__name__)


def f2w(f):
    return 2*np.pi*f


def period(f):
    return 1 / f


def exposure(hz):
    return 1 / hz


def scantime(t, hz):
    return np.linspace(0, t, t*hz)


def sinusoid(A, f, p, t, hz):
    """Continuous"""
    w = f2w(f)
    p = np.mod(p, 2*np.pi)
    return A * np.sin(w*t - p)


def triangle(A, f, p, t, hz):
    """Continuous"""
    a = 0.5 * period(f)
    ts = t - p/(2*np.pi)/f
    q = np.floor(ts/a + 0.5)
    return A * (2/a * (ts - a*q) * np.power(-1, q))


def sawtooth(A, f, p, t, hz):
    """Discontinuous"""
    a = 0.5 * period(f)
    ts = t - p/(2*np.pi)/f
    q = np.floor(ts/a + 0.5)
    return A * (2 * (ts/a - q))


def square(A, f, p, t, hz):
    """Discontinuous"""
    ts = t - p/(2*np.pi)/f
    return A * (np.power(-1, np.floor(2*f*ts)))


def staircase(A, f, p, t, hz):
    """Discontinuous"""
    ts = t - p/(2*np.pi)/f
    return A/f/2 * np.floor(2*f*ts) - A


def lissajous(A, B, fx, fy, px, py, time, hz):
    t = scantime(time, hz)
    x = sinusoid(A, fx, px, t, hz)
    y = sinusoid(B, fy, py, t, hz)
    return x, y, t


def raster(A, B, fx, fy, px, py, time, hz):
    t = scantime(time, hz)
    x = triangle(A, fx, px, t, hz)
    y = staircase(B, fy, py, t, hz)
    return x, y, t


def spiral(A, B, fx, fy, px, py, time, hz):
    t = scantime(time, hz)
    x = sawtooth(A, 0.5*fx, px, t, hz)
    y = sawtooth(B, 0.5*fy, py, t, hz)
    return x, y, t


def scan3(A, B, fx, fy, fz, px, py, time, hz):
    x, y, t = lissajous(A, B, fx, fy, px, py, time, hz)
    z = sawtooth(np.pi, 0.5*fz, 0.5*np.pi, t, hz)
    return x, y, z, t


def avgspeed(time, x, y=None, z=None):
    return distance(x, y, z) / time


def lengths(x, y=None, z=None):
    if y is None:
        y = np.zeros(x.shape)
    if z is None:
        z = np.zeros(x.shape)
    a = np.diff(x)
    b = np.diff(y)
    c = np.diff(z)
    return np.sqrt(a*a + b*b + c*c)


def distance(x, y=None, z=None):
    d = lengths(x, y, z)
    return np.sum(d)


def norm(a):
    """Return the L2 norm of each row"""
    return np.sqrt(np.einsum('ij,ij->i', a, a))


def discrete_trajectory(trajectory, tmin, tmax, dx, dt, max_iter=16):
    """Compute positions along the `trajectory` between `tmin` and `tmax` such
    that space between measurements is never more than `dx` and the time
    between measurements is never more than `dt`.

    Parameters
    ----------
    trajectory : function(time) -> [[x, y, z], [x, y, z]]
        A *continuous* function taking one input and returns a (2, N) vector
        describing the end points of a line segment.
    tmin, tmax : float
        The start and end times.
    dx : float
        The maximum spatial step size.
    dt : float
        The maximum time step size.
    max_iter : int
        The number of attempts to allowed to find a step less than
        `dx` and `dt`.

    Returns
    -------
    position : list of (2 * N) vectors [m]
        Discrete measurement positions along the trajectory satisfying
        constraints.
    dwell : list of float [s]
        The time spent at each position before moving to the next measurement.
    time : list of float [s]
        Discrete times along trajectory satisfying constraints.

    Implementation
    --------------
    Keeping time steps below `dt` for 'trajectory(time)' is trivial, but
    keeping displacement steps below `dx` is not. We use the following
    assumption and proof to ensure that any the probe does not move more than
    `dx` within the area of interest.

    Given that `x` is a point on the line segment `AB` between the endpoints a
    and `b`. Prove that for all affine transformations of `AB`, `AB' = T(AB)`,
    the magnitude of displacement, `dx` is less than or equal to `da` or `db`.

    [Insert proof here.]

    Thus, if at all times, both points used to define the
    probe remains outside the region of interest, then it can be said that
    probe movement within the area of interest is less than dx or equal to dx
    by controlling the movement of the end points of the probe. The users is
    responsible for checking whether the output conforms to this constraint.

    Measurements along the trajectory are generated using a binary search.
    First a starting point is generated, then the time is incremented by `dt`
    is generated, if this point is too farther than `dx` from the previous,
    another point is generated recursively between the previous two.
    """
    position, time, dwell, nextxt = list(), list(), list(), list()
    t, tnext = tmin, tmin + dt
    x = trajectory(t)
    while t <= tmax:
        if not nextxt:
            xnext = trajectory(tnext)
        elif len(nextxt) > max_iter:
            raise RuntimeError("Failed to find next step within {} tries. "
                               "Probably the function is discontinuous."
                               .format(max_iter))
        else:
            xnext, tnext = nextxt.pop()
        if np.all(norm(xnext - x) <= dx):
            position.append(x.flatten())
            time.append(t)
            dwell.append(tnext - t)
            x, t = xnext, tnext
            tnext = t + dt
        else:
            nextxt.append((xnext, tnext))
            tnext = (tnext + t) / 2
            nextxt.append((trajectory(tnext), tnext))

    return position, dwell, time
