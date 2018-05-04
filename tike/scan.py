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

Coverage Workflow
-----------------

1. User defines a function in `thetahv` space. `user_func(t) -> [theta,h,v]`

2. Using the `density_profile` of the probe in the `h, v` plane, create a
    weighted list of thick lines to represent the `Probe`.

3. For each lines, wrap the user function in an offset function.
    `lines_offset(user_func(t)) -> [theta,h,v]`

5. Send the wrapped functions to `discrete_trajectory` to generate a list of
    lines positions which need to be added to the coverage map.

6. Send the list of line positions with their weights to the coverage
    approximator. Weights are used to approximate the density profile of the
    probe.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import logging
from tqdm import tqdm
from tike.tomo import coverage
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
           'scan3',
           'avgspeed',
           'lengths',
           'distance',
           'Probe',
           'discrete_trajectory']


logger = logging.getLogger(__name__)


class Probe(object):
    """Generates procedures for coverage metrics.

    `Probe` moves in a 3D coordinate system: `theta, h, v`. `h, v` are the
    horizontal vertical directions perpendiclar to the probe direction
    where positive directions are to the right and up respectively. `theta` is
    the rotation angle around the vertical reconstruction space axis, `z`. `z`
    is parallel to `v`, and uses the right hand rule to determine
    reconstruction space coordinates `x, y, z`. `theta` is measured from the
    `x` axis, so when `theta = 0`, `h` is parallel to `y`.

    The default probe is a 1 mm^2 square of uniform intensity.

    Attributes
    ----------
    density_profile : function(h, v) -> intensity
        A function that describes the intensity of the beam in the `h, v`
        plane, centered on `[0, 0]`.
    extent : :py:class:`np.array` [cm]
        A rectangle confining the `Probe`. Specify width and aspect ratio.
    """
    def __init__(self, density_profile=None, width=None, aspect=None):
        if density_profile is None:
            self.density_profile = lambda h, v: 1
        else:
            self.density_profile = density_profile
        if width is None:
            self.width = 0.1
            self.aspect = 1
        else:
            self.width = width
            self.aspect = aspect
        self.line_width = 1

    def line_offsets(self, pixel_size, lines_per_pixel=16):
        """Generate a list of ray offsets and weights from density profile
        and extent.

        Returns
        -------
        rays : array [cm]
            [([theta, h, v], intensity_weight), ...]
        lines_per_pixel : int
            Number of lines per pixel_size used to approximate the beam. From a
            previous study, we determined that the this value should be at
            least 16. A higher number will improve the quality of the coverage
            approximation. The number of lines used to approximate the beam
            will be `lines_per_pixel**2`.
        """
        # return [([0, 0, 0], 1), ]  # placeholder
        width, height = self.width, self.width * self.aspect
        # determine if the probe extent is larger than the maximum line_width
        self.line_width = pixel_size/lines_per_pixel
        if width < self.line_width or height < self.line_width:
            # line_width needs to shrink to fit within probe
            raise NotImplementedError("Why are you using such large pixels?")
        else:
            # probe is larger than line_width
            ncells = np.ceil(width / self.line_width)
            self.line_width = width / ncells
        gh = (np.linspace(0, width, int(width/self.line_width), endpoint=False)
              + (self.line_width - width) / 2)
        gv = (np.linspace(0, height, int(height/self.line_width),
                          endpoint=False)
              + (self.line_width - height) / 2)
        offsets = np.meshgrid(gh, gv, indexing='xy')
        h = offsets[0].flatten()
        v = offsets[1].flatten()
        thv = np.stack([np.zeros(v.shape), h, v], axis=1)
        lines = list()
        for row in thv:
            weight = self.density_profile(row[1], row[2])
            if weight > 0:  # Only append lines whose weight matters
                lines.append((row, weight))
        return lines

    def procedure(self, trajectory, pixel_size, tmin, tmax, dt):
        """Return the discrete procedure description from the given trajectory.

        Parameters
        ----------
        trajectory : function(t) -> [theta, h, v]
            A function which describes the position of the center of the Probe
            as a function of time.
        pixel_size : float [cm]
            The edge length of the pixels in the coverage map. Underestimate
            if you don't know.

        Returns
        -------
        procedure : :py:class:`np.array` [radians, cm, ..., s]
            A (M,4) array which describes a series of lines:
            `[theta, h, v, weight]` to approximate the trajectory
        """
        procedure = list()
        trajectory_cache = dict()
        lines = self.line_offsets(pixel_size=pixel_size)
        dx = self.line_width
        # If two or more lines have the same h coordinate, then their discrete
        # trajectories are parallel and only differ in the v direction.
        for offset, weight in tqdm(lines):
            cache_key = tuple(offset[0:2])
            # Check chache for parallel trajectories
            if cache_key in trajectory_cache:
                # compute trajectory from previous
                position, dwell = trajectory_cache[cache_key]
            else:
                th_offset = np.array([offset[0], offset[1], 0])

                def line_trajectory(t):
                    return trajectory(t) + th_offset

                position, dwell, none = discrete_trajectory(
                                            trajectory=line_trajectory,
                                            tmin=tmin, tmax=tmax,
                                            dx=dx, dt=dt)
                trajectory_cache[cache_key] = (position, dwell)

            position = np.concatenate([position,
                                       np.atleast_2d(dwell * weight).T],
                                      axis=1)
            position[:, 2] += offset[2]
            procedure.append(position)
        procedure = np.concatenate(procedure)
        return procedure

    def coverage(self, trajectory, region, pixel_size, tmin, tmax, dt,
                 anisotropy=False):
        """Return a coverage map using this probe.

        The intersection between each line and each pixel is approximated by
        the product of `line_width**2` and the length of segment of the
        line segment `alpha` which passes through the pixel along the line.

        Parameters
        ----------
        trajectory : function(t) -> [theta, h, v] [radians, cm]
            A function which describes the position of the center of the Probe
            as a function of time.
        region : :py:class:`np.array` [cm]
            A box in which to map the coverage. Specify the bounds as
            `[[min_x, max_x], [min_y, max_y], [min_z, max_z]]`.
            i.e. column vectors pointing to the min and max corner.
        pixel_size : float [cm]
            The edge length of the pixels in the coverage map.
        anisotropy : bool
            Whether the coverage map includes anisotropy information. If
            `anisotropy` is `True`, then `coverage_map.shape` is
            `(L, M, N, 2, 2)`, where the two extra dimensions contain coverage
            anisotropy information as a second order tensor.

        Returns
        -------
        coverage_map : :py:class:`numpy.ndarray` [s]
            A discretized map of the approximated procedure coverage.
        """
        if anisotropy:
            raise NotImplementedError
        box = np.asanyarray(region)
        assert np.all(box[:, 0] <= box[:, 1]), ("region minimum must be <= to"
                                                "region maximum.")

        procedure = self.procedure(trajectory=trajectory,
                                   pixel_size=pixel_size,
                                   tmin=tmin, tmax=tmax, dt=dt)

        # Scale to a coordinate system where pixel_size is 1.0
        h = procedure[:, 1] / pixel_size
        v = procedure[:, 2] / pixel_size
        w = procedure[:, 3] * self.line_width**2 / pixel_size**2
        box = box / pixel_size
        # Find new min corner and size of region
        ibox_shape = (np.ceil(box[:, 1] - box[:, 0])).astype(int)
        procedure = np.array(procedure)
        coverage_map = coverage(box[:, 0], ibox_shape,
                                theta=procedure[:, 0], h=h, v=v,
                                line_weight=w)
        return coverage_map


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
    p = np.mod(p, 2*np.pi)
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

    The lissajous is centered on the origin.

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


def raster(A, B, fx, fy, px, py, t):
    """Return the value of a raster function at time `t`.
    #discontinuous #2d

    The raster starts at the origin and moves initially in the positive
    directions. `fy` should be `2*fx` to make a conventional raster.

    Parameters
    A : float
        The maximum horizontal displacement of the function at half the period.
        Every period the horizontal displacement is 0.
    B : float
        The maximum vertical displacement every period.
    fx, fy : float
        The temporal frequencies of the function.
    px, py : float
        The phase shifts of the x and y components of the function
    """
    x = triangle(A, fx, px+np.pi/2, t) + A
    y = staircase(B, fy, py, t)
    return x/2, y


def spiral(A, B, fx, fy, px, py, t):
    """Return the value of a spiral function at time `t`.
    #discontinuous #2d

    The spiral is centered on the origin.

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
    return sqrt(a*a + b*b + c*c)


def distance(x, y=None, z=None):
    d = lengths(x, y, z)
    return np.sum(d)


def euclidian_dist(a, b, r=0.75):
    """Return the euclidian distance between two points in theta, h, v space.

    Parameters
    ----------
    a, b : (3, ) array-like
        The two points to find the distance between
    r : float
        The radius to use when converting to euclidian space.
    """
    r1 = sqrt(r**2 + a[1]**2)
    r2 = sqrt(r**2 + b[1]**2)
    theta1 = a[0] + atan2(a[1], r)
    theta2 = b[0] + atan2(b[1], r)
    cosines = r1**2 + r2**2 - 2*r1*r2*cos(theta1-theta2)
    return sqrt(abs(cosines) + (a[2]-b[2])**2)


def euclidian_dist_approx(a, b, r=0.75):
    """Approximate the euclidian distance by adding the arclength to the
       hv distance. It's about 2x faster than the unapproximated.
    """
    return abs(a[0]-b[0]) * r + sqrt((a[1]-b[1])**2 + (a[2]-b[2])**2)


def discrete_trajectory(trajectory, tmin, tmax, dx, dt, max_iter=16):
    """Compute positions along the `trajectory` between `tmin` and `tmax` such
    that space between measurements is never more than `dx` and the time
    between measurements is never more than `dt`.

    Parameters
    ----------
    trajectory : function(time) -> [theta, h, v]
        A *continuous* function taking one input and returns a (N,) vector
        describing position of the line.
    [tmin, tmax) : float
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
    position : list of (N,) vectors [m]
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

    [TODO: Insert proof here.]

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
    t, tnext = tmin, min(tmin + dt, tmax)
    x = trajectory(t)
    while t < tmax:
        if not nextxt:
            xnext = trajectory(tnext)
        elif len(nextxt) > max_iter:
            raise RuntimeError("Failed to find next step within {} tries. "
                               "Probably the function is discontinuous."
                               .format(max_iter))
        else:
            xnext, tnext = nextxt.pop()
        if euclidian_dist_approx(xnext, x) <= dx:
            position.append(x)
            time.append(t)
            dwell.append(tnext - t)
            x, t = xnext, tnext
            tnext = min(t + dt, tmax)
        else:
            nextxt.append((xnext, tnext))
            tnext = (tnext + t) / 2
            nextxt.append((trajectory(tnext), tnext))

    return position, dwell, time
