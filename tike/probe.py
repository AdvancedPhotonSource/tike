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
           'discrete_trajectory',
           'coded_exposure',
           ]


logging.basicConfig(level=logging.INFO)
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
    width : float [cm]
        The width of the density_profile.
    aspect : float
        height / width ratio of density_profile
    lines_per_cm : int
        Number of lines per centimeter used to approximate the beam. From a
        previous study, we determined that the this value should be at
        least 16 lines per voxel width. A higher number will improve the
        quality of the coverage approximation.
    """
    def __init__(self, density_profile=None, width=0.1, aspect=1.0,
                 lines_per_cm=16):
        if density_profile is None:
            self.density_profile = lambda h, v: 1.0
        else:
            self.density_profile = density_profile
        assert width > 0.0
        assert aspect > 0.0
        self.width = width
        self.aspect = aspect
        assert lines_per_cm > 0.0
        self.lines_per_cm = lines_per_cm
        self.line_width = 1. / lines_per_cm

    def line_offsets(self, pixel_size=None):
        """Generate a list of ray offsets and weights from density profile
        and extent.

        Returns
        -------
        lines : [([theta, h, v], intensity_weight), ...] [cm]
            A list of tuples where the first element in the tuple is the offset
            in theta, h, v, coordinates and the second elemetn is the weight
            of the line according to self.density_profile.
        """
        # return [([0, 0, 0], 1), ]  # placeholder
        width, height = self.width, self.width * self.aspect
        # determine if the probe extent is larger than the maximum line_width
        self.line_width = 1. / self.lines_per_cm
        if width < self.line_width or height < self.line_width:
            # line_width needs to shrink to fit within probe
            raise NotImplementedError("lines_per_cm is too small!")
        else:
            # probe is larger than line_width
            ncells = np.ceil(width / self.line_width)
            self.line_width = width / ncells
        logger.info(" line_width is {:n} cm".format(self.line_width))
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
        logger.info(" probe uses {:,d} lines".format(len(lines)))
        return lines

    def procedure(self, trajectory, tmin, tmax, tstep, tkwargs={}):
        """Return the discrete procedure description from the given trajectory.

        Parameters
        ----------
        trajectory : function(t, **tkwargs) -> theta, h, v
            A function which describes the position of the center of the Probe
            as a function of time.

        Returns
        -------
        theta, h, v, dwell, time : :py:class:`np.array` [radians, cm, ..., s]
            A (M,) array which describes a series of lines to approximate the
            trajectory
        """
        all_theta, all_h, all_v = list(), list(), list()
        all_dwell, all_time = list(), list()
        trajectory_cache = dict()
        lines = self.line_offsets()
        xstep = self.line_width
        # If two or more lines have the same h coordinate, then their discrete
        # trajectories are parallel and only differ in the v direction.
        for offset, weight in lines:
            cache_key = tuple(offset[0:2])
            # Check chache for parallel trajectories
            if cache_key in trajectory_cache:
                # compute trajectory from previous
                th, h, v, dwell, time = trajectory_cache[cache_key]
            else:
                def line_trajectory(t, **tkwargs):
                    pos_temp = trajectory(t, **tkwargs)
                    return (pos_temp[0] + offset[0],
                            pos_temp[1] + offset[1],
                            pos_temp[2])
                th, h, v, dwell, time = discrete_trajectory(
                                            trajectory=line_trajectory,
                                            tkwargs=tkwargs,
                                            tmin=tmin, tmax=tmax,
                                            xstep=xstep, tstep=tstep)
                trajectory_cache[cache_key] = (th, h, v, dwell, time)
            all_theta.append(th)
            all_h.append(h)
            all_v.append(v + offset[2])
            all_dwell.append(dwell * weight)
            all_time.append(time)
        all_theta = np.concatenate(all_theta)
        logger.info(" procedure is {:,d} lines".format(all_theta.size))
        return (all_theta, np.concatenate(all_h),
                np.concatenate(all_v), np.concatenate(all_dwell),
                np.concatenate(all_time))

    def coverage(self, trajectory, region, pixel_size, tmin, tmax, tstep,
                 anisotropy=1, tkwargs={}):
        """Return a coverage map using this probe.

        The intersection between each line and each pixel is approximated by
        the product of `line_width**2` and the length of segment of the
        line segment `alpha` which passes through the pixel along the line.

        Parameters
        ----------
        trajectory : function(t, **tkwargs) -> theta, h, v [radians, cm]
            A function which describes the position of the center of the Probe
            as a function of time.
        region : :py:class:`np.array` [cm]
            A box in which to map the coverage. Specify the bounds as
            `[[min_z, max_z], [min_x, max_x], [min_y, max_y]]`.
            i.e. column vectors pointing to the min and max corner.
        pixel_size : float [cm]
            The edge length of the pixels in the coverage map.
        anisotropy : int
            The number of angle bins to include in the coverage map. If
            `anisotropy` is `O > 1`, then `coverage_map.shape` is
            `(L, M, N, O)`, where the magnitude of coverage from each of the
            `O` directions is stored separately.

        Returns
        -------
        coverage_map : :py:class:`numpy.ndarray` [s]
            A discretized map of the approximated procedure coverage.
        """
        box = np.asanyarray(region)
        assert np.all(box[:, 0] <= box[:, 1]), ("region minimum must be <= to"
                                                "region maximum.")

        theta, h, v, w, t = self.procedure(trajectory=trajectory,
                                           tkwargs=tkwargs,
                                           tmin=tmin, tmax=tmax, tstep=tstep)

        w = w * self.line_width**2 / pixel_size**3
        # Find new min corner and size of region
        ibox_shape = box[:, 1] - box[:, 0]
        ngrid = np.ceil(ibox_shape / pixel_size).astype(np.int)
        coverage_map = coverage(box[:, 0], ibox_shape, ngrid,
                                theta=theta, h=h, v=v,
                                line_weight=w, anisotropy=anisotropy)
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


def discrete_trajectory(trajectory, tmin, tmax, xstep, tstep, tkwargs={}):
    """Create a linear approximation of `trajectory` between `tmin` and `tmax`
    such that space between measurements is less than `xstep` and the time
    between measurements is less than `tstep`.

    Parameters
    ----------
    trajectory : function(time, **tkwargs) -> theta, h, v
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
    theta, h, v : (N,) vectors [m]
        Discrete measurement positions along the trajectory satisfying
        constraints.
    dwell : (N,) vector [s]
        The time spent at each position before moving to the next measurement.
    time : (N,) vector [s]
        Discrete times along trajectory satisfying constraints.
    """
    dist_func = euclidian_dist_approx
    all_theta, all_h, all_v, all_times = discrete_helper(trajectory,
                                                         tmin, tmax,
                                                         xstep, tstep,
                                                         dist_func,
                                                         tkwargs=tkwargs)
    all_theta = np.concatenate(all_theta)
    all_h = np.concatenate(all_h)
    all_v = np.concatenate(all_v)
    all_times = np.concatenate(all_times)
    # Compute dwell time because you can't while recursing
    dwell = np.empty(all_times.size)
    dwell[0:-1] = np.diff(all_times)
    dwell[-1] = tmax - all_times[-1]
    assert tmax - all_times[-1] <= tstep, "Last time not less than tstep"
    assert np.all(dwell <= tstep + 1e-6), \
        "Some times not less than tstep\n{}".format(dwell[dwell > tstep][0])
    # assert dist_func([trajectory(all_times[-1]), trajectory(tmax)]) < xstep,
    #     "Last distance not less than dstep"
    assert np.all(dist_func(all_theta, all_h, all_v) <= xstep), \
        "Some distances wrong"
    # These assertions are vulnerable to rounding errors
    return all_theta, all_h, all_v, dwell, all_times


def discrete_helper(trajectory, tmin, tmax, xstep, tstep, dist_func,
                    tkwargs={}):
    """Do a recursive sampling of the trajectory."""
    all_theta, all_h = list(), list()
    all_v, all_times = list(), list()
    # Sample en masse the trajectory over time
    times = np.arange(tmin, tmax + tstep, tstep)
    theta, h, v = trajectory(times, **tkwargs)
    # Compute spatial distances between samples
    distances = dist_func(theta, h, v)
    # determine which ranges are too large and which to keep
    keepit = xstep > distances
    len_keepit = keepit.size
    rlo, rhi, klo, khi = 0, 0, 0, 0
    while khi < len_keepit:
        khi += 1
        rhi += 1
        if not keepit[klo]:
            klo += 1
        elif khi == len_keepit or not keepit[rhi]:
            # print("keep: {}, {}".format(klo, khi))
            # concatenate the ranges to keep
            all_theta.append(theta[klo:khi])
            all_h.append(h[klo:khi])
            all_v.append(v[klo:khi])
            all_times.append(times[klo:khi])
            klo = khi
        if keepit[rlo]:
            rlo += 1
        elif rhi == len_keepit or keepit[rhi]:
            # print("replace: {}, {} with {} tstep".format(rlo, rhi, tstep/2))
            # concatenate the newly calculated region
            itheta, ih, iv, itimes = discrete_helper(trajectory,
                                                     times[rlo], times[rhi],
                                                     xstep, tstep/2,
                                                     dist_func, tkwargs)
            all_theta += itheta
            all_h += ih
            all_v += iv
            all_times += itimes
            rlo = rhi
    khi += 1
    return all_theta, all_h, all_v, all_times


def coded_exposure(theta, h, v, time, dwell, c_time, c_dwell):
    """Returns the intersection of a scanning procedure and coded exposure
    with measurements reordered and bundled by code.

    Given a series of discrete measurements with time and duration
    (dwell) and series of coded exposures, the measurments are adjusted to only
    include measurements that fit under the masks. The measurements are also
    sorted by which code they fit into.

    Measurements and codes must be ordered monotonically increasing by time
    i.e. time[1] >= time[0].

    Essentially this function bins the measurements into the time codes, but if
    a measurement covers multiple bins, then it is put into all of them.

    Parameters
    ----------
    theta, h, v : :py:class:`numpy.array` (M, )
        The position of each ray at each measurement.
    dwell, time : :py:class:`numpy.array` (M, )
        The duration and start time of each measurement.
    c_time, c_dwell :py:class:`numpy.array` (N, )
        The start and extent of each exposure.

    Returns
    -------
    theta1, h1, v1, time1, dwell1 :py:class:`numpy.array` (M, )
        New position and time coordates which fit into the code.
    bundles : :py:class:`numpy.array` (N, )
        The starting index of each coded bundle.
    """
    _fname = "coded_exposure"
    # Implementation uses the assumption that both the measurement times
    # and coded times are monotonically increasing in order to generate the
    # intersection faster than a binary search tree
    assert(monotonic(time))
    assert(monotonic(c_time))
    # Check if any of the codes overlap with measurements
    if not has_overlap(time[0], dwell[-1] + time[-1] - time[0],
                       c_time[0], c_dwell[-1] + c_time[-1] - c_time[0]):
        raise ValueError("Codes don't overlap measurements.")
        return list(), list(), list(), list(), list(), list()

    # Find overlaps of individuals
    start = 0  # Start searching here for the next code overlap
    times = list()
    dwells = list()  # store new coded times
    positions = list()
    codes = list()
    for measurement in range(0, time.size):
        found_atleast_one = False
        logging.debug("{}: Measurement {}".format(_fname, measurement))
        for code in range(start, c_time.size):
            logging.debug("{}: Checking code {}".format(_fname, code))
            if has_overlap(time[measurement], dwell[measurement],
                           c_time[code], c_dwell[code]):
                # Record the intersection
                t1, d1 = get_overlap(time[measurement], dwell[measurement],
                                     c_time[code], c_dwell[code])
                if d1 > 0:
                    logging.debug("{}: Overlap found: {}, {}".format(_fname,
                                                                     t1, d1))
                    codes.append(code)
                    positions.append(measurement)
                    times.append(t1)
                    dwells.append(d1)
                    if not found_atleast_one:
                        found_atleast_one = True
                        # Always start searching at the start of last known
                        # overlap
                        start = code
            elif found_atleast_one:
                # This code is the one after the overlap
                break
    # Reorder results to bundle all measurements within the same code
    new_order = np.argsort(codes)
    codes = np.array(codes)[new_order]
    positions = np.array(positions)[new_order]
    times1 = np.array(times)[new_order]
    dwells1 = np.array(dwells)[new_order]
    # Clip the measurements
    bundles = np.nonzero(np.diff(np.concatenate([[-1], codes])))[0]
    return (theta[positions], h[positions], v[positions], times1, dwells1,
            bundles)


def monotonic(x):
    """Check whether x is monomtically increasing"""
    dx = np.diff(x)
    return np.all(dx >= 0)


def has_overlap(x0, xd, y0, yd):
    """Return True if ranges overlap

    Parameters
    ----------
    x0, y0 : float
        The min values of the ranges
    xd, yd : float
        The widths of the ranges
    """
    return x0 + xd >= y0 and y0 + yd >= x0


def get_overlap(x0, xd, y0, yd):
    """Return the min edge and width of overlap

    Parameters
    ----------
    x0, y0 : float
        The min values of the ranges
    xd, yd : float
        The widths of the ranges

    Returns
    -------
    lo : float
        The min value of the overlap region
    width : float
        The width of the overlap region. May be zero if ranges share an edge.
    """
    lo = max(x0, y0)
    width = min(x0 + xd, y0 + yd) - lo
    assert width >= 0, "These two ranges don't actually overlap"
    return lo, width
