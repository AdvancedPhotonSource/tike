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
    weighted list of thick rays to represent the `Probe`.

3. For each ray, wrap the user function in an offset function.
    `ray_offset(user_func(t)) -> [theta,h,v]`

4. Wrap the ray offset functions in the `vhtheta_to_xyz` converter.
    This function also adds an additional point to the output.
    `vhtheta_to_xyz(ray_offset(t)) -> [[xyz], [xyz]]`

5. Send the wrapped functions to `discrete_trajectory` to generate a list of
    ray positions which need to be added to the coverage map.

6. Send the list of ray positions with their weights to the coverage
    approximator. Weights are used to approximate the density profile of the
    probe.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import logging
import warnings
from tqdm import tqdm

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

    def line_offsets(self, pixel_size, N=1):
        """Generate a list of ray offsets and weights from density profile
        and extent.

        From a previous study, we determined that the `line_width` should be at
        most 1/16 of the pixel size.

        Returns
        -------
        rays : array [cm]
            [([theta, h, v], intensity_weight), ...]
        """
        # return [([0, 0, 0], 1), ]  # placeholder
        width, height = self.width, self.width * self.aspect
        # determine if the probe extent is larger than the maximum line_width
        self.line_width = pixel_size/16
        if width < self.line_width or height < self.line_width:
            # line_width needs to shrink to fit within probe
            raise NotImplementedError("Why are you using such large pixels?")
        else:
            # probe is larger than line_width
            ncells = np.ceil(width / self.line_width)
            self.line_width = width / ncells
        gh = np.linspace(0, width, int(width/self.line_width)) - width/2
        gv = np.linspace(0, height, int(height/self.line_width)) - height/2
        offsets = np.meshgrid(gh, gv, indexing='xy')
        h = offsets[0].flatten()
        v = offsets[1].flatten()
        thv = np.stack([np.zeros(v.shape), h, v], axis=1)
        lines = list()
        for row in thv:
            lines.append((row, self.density_profile(row[1], row[2])))
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
        procedure : list of :py:class:`np.array` [cm, ..., s]
            A (M,7) array which describes a series of lines:
            `[x1, y1, z1, x2, y2, z2, weight]` to approximate the trajectory
        """
        positions = list()
        lines = self.line_offsets(pixel_size=pixel_size)
        dx = self.line_width
        # TODO: Each iteration of this loop is a separate line. If two or more
        # lines have the same h coordinate, then their discrete trajectories
        # parallel and offset in the v direction. #optimization
        for offset, weight in tqdm(lines):
            def line_trajectory(t):
                return thetahv_to_xyz(trajectory(t) + offset)
            position, dwell, none = discrete_trajectory(trajectory=line_trajectory,
                                                        tmin=tmin, tmax=tmax,
                                                        dx=dx, dt=dt)
            position = np.concatenate([position,
                                       np.atleast_2d(dwell * weight).T],
                                      axis=1)
            positions.append(position)
        return positions

    def coverage(self, trajectory, region, pixel_size, tmin, tmax, dt,
                 anisotropy=False):
        """Return a coverage map using this probe.

        trajectory : function(t) -> [theta, h, v] [radians, cm]
            A function which describes the position of the center of the Probe
            as a function of time.
        region : :py:class:`np.array` [cm]
            A box in which to map the coverage. Specify the bounds as
            `[[min_x, max_x], [min_y, max_y], [min_z, max_z]]`.
            i.e. column vectors pointing to the min and max corner.
        pixel_size : float [cm]
            The edge length of the pixels in the coverage map.
        """
        procedure = self.procedure(trajectory=trajectory,
                                   pixel_size=pixel_size, tmin=tmin, tmax=tmax,
                                   dt=dt)
        procedure = np.concatenate(procedure)
        return coverage_approx(procedure=procedure, region=region,
                               pixel_size=pixel_size,
                               line_width=self.line_width,
                               anisotropy=anisotropy)


def coverage_approx(procedure, region, pixel_size, line_width,
                    anisotropy=False):
    """Approximate procedure coverage with thick lines.

    The intersection between each line and each pixel is approximated by
    the product of `line_width**2` and the length of segment of the
    line segment `alpha` which passes through the pixel along the line.

    If `anisotropy` is `True`, then `coverage_map.shape` is `(L, M, N, 2, 2)`,
    where the two extra dimensions contain coverage anisotropy information as a
    second order tensor.

    Parameters
    ----------
    procedure : list of :py:class:`np.array` [cm, ..., s]
        Each element of 'procedure' is a (7,) array which describes a series
        of lines as `[x1, y1, z1, x2, y2, z2, weight]`. Presently, `z1`
        must equal `z2`.
    line_width` : float [cm]
        The side length of the square cross-section of the line.
    region : :py:class:`np.array` [cm]
        A box in which to map the coverage. Specify the bounds as
        `[[min_x, max_x], [min_y, max_y], [min_z, max_z]]`.
        i.e. column vectors pointing to the min and max corner.
    pixel_size : float [cm]
        The edge length of the pixels in the coverage map in centimeters.
    anisotropy : bool
        Determines whether the coverage map includes anisotropy information.

    Returns
    -------
    coverage_map : :py:class:`numpy.ndarray` [s]
        A discretized map of the approximated procedure coverage.

    Raises
    ------
    ValueError when lines have have non-constant z coordinate.
    """
    box = np.asanyarray(region)
    # Define the locations of the grid lines (gx, gy, gz)
    gx = np.arange(box[0, 0], box[0, 1] + pixel_size, pixel_size)
    gy = np.arange(box[1, 0], box[1, 1] + pixel_size, pixel_size)
    gz = np.arange(box[2, 0], box[2, 1] + pixel_size, pixel_size)
    # the number of pixels = number of gridlines - 1
    sx, sy, sz = gx.size-1, gy.size-1, gz.size-1
    # Preallocate the coverage_map
    if anisotropy:
        coverage_map = np.zeros((sx, sy, sz, 2, 2))
    else:
        coverage_map = np.zeros((sx, sy, sz))
    for line in tqdm(procedure):
        # line -> x1, y1, z1, x2, y2, z2, weight
        x0, y0, z0 = line[0], line[1], line[2]
        x1, y1, z1 = line[3], line[4], line[5]
        weight = line[6]
        # avoid upper-right boundary errors
        if (x1 - x0) == 0:
            x0 += 1e-12
        if (y1 - y0) == 0:
            y0 += 1e-12
        if (z1 - z0) != 0:
            raise ValueError("Lines must have constant z coordinate.")
        # vector lengths (ax, ay)
        ax = (gx - x0) / (x1 - x0)
        ay = (gy - y0) / (y1 - y0)
        # layer weights in the z direction
        az = np.maximum(0, (np.minimum(z1 + line_width/2, (gz + pixel_size))
                            - np.maximum(z0 - line_width/2, gz)) / line_width)
        # edges of alpha (a0, a1)
        ax0 = min(ax[0], ax[-1])
        ax1 = max(ax[0], ax[-1])
        ay0 = min(ay[0], ay[-1])
        ay1 = max(ay[0], ay[-1])
        a0 = max(max(ax0, ay0), 0)
        a1 = min(min(ax1, ay1), 1)
        # sorted alpha vector
        cx = (ax >= a0) & (ax <= a1)
        cy = (ay >= a0) & (ay <= a1)
        alpha = np.sort(np.r_[ax[cx], ay[cy]])
        if len(alpha) > 0:
            # lengths
            xv = x0 + alpha * (x1 - x0)
            yv = y0 + alpha * (y1 - y0)
            lx = np.ediff1d(xv)
            ly = np.ediff1d(yv)
            dist = np.sqrt(lx**2 + ly**2)
            dist2 = np.dot(dist, dist)
            ind = dist != 0
            # indexing
            mid = alpha[:-1] + np.ediff1d(alpha) / 2.
            xm = x0 + mid * (x1 - x0)
            ym = y0 + mid * (y1 - y0)
            ix = np.floor(np.true_divide(sx * (xm - box[0, 0]),
                                         sx * pixel_size)).astype('int')
            iy = np.floor(np.true_divide(sy * (ym - box[0, 1]),
                                         sy * pixel_size)).astype('int')
            iz = np.arange(sz + 1)[az > 0].astype('int')
            try:
                magnitude = dist * line_width**2 * weight
                if anisotropy:
                    beam = line[0:2] - line[3:5]
                    beam_angle = np.arctan2(beam[1], beam[0])
                    tensor = tensor_at_angle(beam_angle, magnitude)
                    for i in iz:
                        coverage_map[ix, iy, i, :, :] += tensor * az[i]
                else:
                    for i in iz:
                        coverage_map[ix, iy, i] += magnitude * az[i]
            except IndexError as e:
                warnings.warn("{}\nix is {}\niy is {}\niz is {}".format(e, ix,
                              iy, iz), RuntimeWarning)
    return coverage_map / pixel_size**3


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


def thetahv_to_xyz(thv_coords, radius=1.5):
    """Convert `theta, h, v` coordinates to `x, y, z` coordinates.

    Parameters
    ----------
    thv_coords : :py:class:`np.array` [radians, cm, cm]
        The coordinates in `theta, h, v` space.
    radius : float [cm]
        The radius used to place the `h, v` plane in `x, y, z` space.
    """
    R, theta = np.eye(3), thv_coords[0]
    R[0, 0] = np.cos(theta)
    R[0, 1] = np.sin(theta)
    R[1, 0] = -np.sin(theta)
    R[1, 1] = np.cos(theta)
    return np.dot(np.array([[radius, thv_coords[1], thv_coords[2]],
                           [-radius, thv_coords[1], thv_coords[2]]]), R)
