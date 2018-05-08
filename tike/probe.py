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
Define a Probe class for generating 3D coverage maps of user defined function.

User defines a function in `thetahv` space. `user_func(t) -> theta, h, v`
where t, theta, h, and v are 1D numpy arrays.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import logging
from math import sqrt, atan2, cos
from tike.tomo import coverage

__author__ = "Doga Gursoy, Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['Probe',
           'discrete_trajectory']


logger = logging.getLogger(__name__)


class Probe(object):
    """Generates procedures for coverage metrics.

    `Probe` moves in a 3D coordinate system: `theta, h, v`. `h, v` are the
    horizontal vertical directions perpendiclar to the probe direction
    where positive directions are to the right and up respectively. `theta` is
    the rotation angle around the vertical reconstruction space axis, `z`. `z`
    is parallel to `v`, and uses the right hand rule to determine
    reconstruction space coordinates `z, x, y`. `theta` is measured from the
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

    def procedure(self, trajectory, pixel_size, tmin, tmax, tstep):
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
        xstep = self.line_width
        # If two or more lines have the same h coordinate, then their discrete
        # trajectories are parallel and only differ in the v direction.
        for offset, weight in lines:
            cache_key = tuple(offset[0:2])
            # Check chache for parallel trajectories
            if cache_key in trajectory_cache:
                # compute trajectory from previous
                position, dwell = trajectory_cache[cache_key]
            else:
                def line_trajectory(t):
                    pos_temp = trajectory(t)
                    return (pos_temp[0] + offset[0],
                            pos_temp[1] + offset[1],
                            pos_temp[2])

                position, dwell, none = discrete_trajectory(
                                            trajectory=line_trajectory,
                                            tmin=tmin, tmax=tmax,
                                            xstep=xstep, tstep=tstep)
                trajectory_cache[cache_key] = (position, dwell)

            position = np.stack(position + [dwell * weight], axis=1)
            position[:, 2] += offset[2]
            procedure.append(position)
        procedure = np.concatenate(procedure)
        return procedure

    def coverage(self, trajectory, region, pixel_size, tmin, tmax, tstep,
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
            `[[min_z, max_z], [min_x, max_x], [min_y, max_y]]`.
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
                                   tmin=tmin, tmax=tmax, tstep=tstep)

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


def euclidian_dist(theta, h, v, r=0.75):
    """Return the euclidian distance between consecutive points in
    theta, h, v space.

    Parameters
    ----------
    theta, h, v : (M,) :py:class:`numpy.array`
        Coordinates of points.
    r : float
        The radius to use when converting to euclidian space.
    """
    raise NotImplementedError()
    r1 = sqrt(r**2 + h**2)
    theta1 = np.diff(theta + np.atan2(h, r))
    cosines = r1**2 + r2**2 - 2*r1*r2*cos(theta1)
    return np.sqrt(np.abs(cosines) + v1**2)


def euclidian_dist_approx(theta, h, v, r=0.75):
    """Approximate the euclidian distance between consecutive elements of the
    points by adding the arclength travelled to the hv distance.

    This method is abour 2x faster than the unapproximated. The output array
    size is one less than the input sizes.

    Parameters
    ----------
    theta, h, v : (M,) :py:class:`numpy.array`
        Coordinates of points.
    r : float
        The radius to use when converting to euclidian space.
    """
    t1 = np.diff(theta)
    h1 = np.diff(h)
    v1 = np.diff(v)
    return np.abs(t1) * r + np.sqrt(h1**2 + v1**2)


def discrete_trajectory(trajectory, tmin, tmax, xstep, tstep):
    """Create a linear approximation of `trajectory` between `tmin` and `tmax`
    such that space between measurements is less than `xstep` and the time
    between measurements is less than `tstep`.

    Parameters
    ----------
    trajectory : function(time) -> theta, h, v
        A *continuous* function taking a single 1D array and returning three
        1D arrays.
    [tmin, tmax) : float
        The start and end times.
    xstep : float
        The maximum spatial step size.
    tstep : float
        The maximum time step size.

    Returns
    -------
    position : list of 3 (N,) vectors [m]
        Discrete measurement positions along the trajectory satisfying
        constraints.
    dwell : (N,) vector [s]
        The time spent at each position before moving to the next measurement.
    time : (N,) vector [s]
        Discrete times along trajectory satisfying constraints.
    """
    all_theta, all_h, all_v, all_times = discrete_helper(trajectory,
                                                         tmin, tmax,
                                                         xstep, tstep)
    dwell = np.empty(all_times.size)
    dwell[0:-1] = np.diff(all_times)
    dwell[-1] = tmax - all_times[-1]

    return [all_theta, all_h, all_v], dwell, all_times


def discrete_helper(trajectory, tmin, tmax, xstep, tstep):
    """Do a recursive sampling of the trajectory."""
    all_theta, all_h = np.array([]), np.array([])
    all_v, all_times = np.array([]), np.array([])
    # Sample en masse the trajectory over time
    times = np.arange(tmin, tmax + tstep, tstep)
    theta, h, v = trajectory(times)
    # Compute spatial distances between samples
    distances = euclidian_dist_approx(theta, h, v)
    # determine which ranges are too large and which to keep
    keepit = xstep > distances
    rlo, rhi, klo, khi = 0, 0, 0, 0
    while khi < len(keepit):
        khi += 1
        rhi += 1
        if not keepit[klo]:
            klo += 1
        elif khi == len(keepit) or not keepit[rhi]:
            # print("keep: {}, {}".format(klo, khi))
            # concatenate the ranges to keep
            all_theta = np.concatenate([all_theta, theta[klo:khi]])
            all_h = np.concatenate([all_h, h[klo:khi]])
            all_v = np.concatenate([all_v, v[klo:khi]])
            all_times = np.concatenate([all_times, times[klo:khi]])
            klo = khi
        if keepit[rlo]:
            rlo += 1
        elif rhi == len(keepit) or keepit[rhi]:
            # print("replace: {}, {} with {} tstep".format(rlo, rhi, tstep/2))
            # concatenate the newly calculated region
            itheta, ih, iv, itimes = discrete_helper(trajectory,
                                                     times[rlo], times[rhi],
                                                     xstep, tstep/2)
            all_theta = np.concatenate([all_theta, itheta])
            all_h = np.concatenate([all_h, ih])
            all_v = np.concatenate([all_v, iv])
            all_times = np.concatenate([all_times, itimes])
            rlo = rhi
    khi += 1
    return all_theta, all_h, all_v, all_times
