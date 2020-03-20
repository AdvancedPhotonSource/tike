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
"""Define functions for modifying trajectories."""

__author__ = "Doga Gursoy, Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    'discrete_trajectory',
    'coded_exposure',
]

import logging
import numpy as np

logger = logging.getLogger(__name__)


def euclidian_dist(theta, v, h, r=0.5):
    """Return the euclidian distance between consecutive points.

    Parameters
    ----------
    theta, v, h : (M,) :py:class:`numpy.array`
        Coordinates of points.
    r : float
        The radius to use when converting to euclidian space.

    """
    # Assume the arclength is the same as the tangential displacement
    dr = np.diff(theta) * r
    # Compute the horizontal and vertical components of displacement
    dv = np.diff(v)
    dh = np.abs(np.diff(h)) + np.abs(dr * np.cos(theta))
    # Combine displacement components, ignoring component along beam
    return np.sqrt(dv * dv + dh * dh)


def euclidian_dist_approx(theta, v, h, r=0.75):
    """Approximate the euclidian distance between consecutive elements.

    Approximate by adding the arclength travelled to the vh distance. This
    method is about 2x faster than the unapproximated. The output array
    size is one less than the input sizes.

    Parameters
    ----------
    theta, v, h : (M,) :py:class:`numpy.array`
        Coordinates of points.
    r : float
        The radius to use when converting to euclidian space.

    """
    t1 = np.diff(theta)
    v1 = np.diff(v)
    h1 = np.diff(h)
    return np.abs(t1) * r + np.sqrt(v1**2 + h1**2)


def discrete_trajectory(trajectory, tmin, tmax, xstep, tstep, tkwargs=None):
    """Create a linear approximation of `trajectory` between `tmin` and `tmax`.

    The space between measurements is less than `xstep` and the time
    between measurements is less than `tstep`.

    Parameters
    ----------
    trajectory : function(time, **tkwargs) -> theta, v, h
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
    theta, v, h : (N,) vectors [m]
        Discrete measurement positions along the trajectory satisfying
        constraints.
    dwell : (N,) vector [s]
        The time spent at each position before moving to the next measurement.
    time : (N,) vector [s]
        Discrete times along trajectory satisfying constraints.

    """
    tkwargs = dict() if tkwargs is None else tkwargs
    dist_func = euclidian_dist_approx
    all_theta, all_v, all_h, all_times = discrete_helper(
        trajectory, tmin, tmax, xstep, tstep, dist_func, tkwargs=tkwargs)
    all_theta = np.concatenate(all_theta)
    all_v = np.concatenate(all_v)
    all_h = np.concatenate(all_h)
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
    assert np.all(dist_func(all_theta, all_v, all_h) <= xstep), \
        "Some distances wrong"
    # These assertions are vulnerable to rounding errors
    return all_theta, all_v, all_h, dwell, all_times


def discrete_helper(
        trajectory, tmin, tmax, xstep, tstep, dist_func,
        tkwargs=None
):  # yapf: disable
    """Do a recursive sampling of the trajectory."""
    tkwargs = dict() if tkwargs is None else tkwargs
    all_theta, all_v = list(), list()
    all_h, all_times = list(), list()
    # Sample en masse the trajectory over time
    times = np.arange(tmin, tmax + tstep, tstep)
    theta, v, h = trajectory(times, **tkwargs)
    # Compute spatial distances between samples
    distances = dist_func(theta, v, h)
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
            all_v.append(v[klo:khi])
            all_h.append(h[klo:khi])
            all_times.append(times[klo:khi])
            klo = khi
        if keepit[rlo]:
            rlo += 1
        elif rhi == len_keepit or keepit[rhi]:
            # print("replace: {}, {} with {} tstep".format(rlo, rhi, tstep/2))
            # concatenate the newly calculated region
            itheta, ih, iv, itimes = discrete_helper(
                trajectory, times[rlo], times[rhi], xstep, tstep / 2, dist_func,
                tkwargs)
            all_theta += itheta
            all_v += iv
            all_h += ih
            all_times += itimes
            rlo = rhi
    khi += 1
    return all_theta, all_v, all_h, all_times


def coded_exposure(theta, v, h, time, dwell, c_time, c_dwell):
    """Return the intersection of a scanning procedure and coded exposure.

    Given a series of discrete measurements with time and duration
    (dwell) and series of coded exposures, the measurments are adjusted to only
    include measurements that fit under the masks. The measurements are also
    reordered and bundled by which code they fit into.

    Measurements and codes must be ordered monotonically increasing by time
    i.e. time[1] >= time[0].

    Essentially this function bins the measurements into the time codes, but if
    a measurement covers multiple bins, then it is put into all of them.

    Parameters
    ----------
    theta, v, h : :py:class:`numpy.array` (M, )
        The position of each ray at each measurement.
    dwell, time : :py:class:`numpy.array` (M, )
        The duration and start time of each measurement.
    c_time, c_dwell :py:class:`numpy.array` (N, )
        The start and extent of each exposure.

    Returns
    -------
    theta1, v1, h1, time1, dwell1 :py:class:`numpy.array` (M, )
        New position and time coordates which fit into the code.
    bundles : :py:class:`numpy.array` (N, )
        The starting index of each coded bundle.

    """
    _fname = "coded_exposure"
    # Implementation uses the assumption that both the measurement times
    # and coded times are monotonically increasing in order to generate the
    # intersection faster than a binary search tree
    assert (monotonic(time))
    assert (monotonic(c_time))
    # Check if any of the codes overlap with measurements
    if not has_overlap(time[0], dwell[-1] + time[-1] - time[0], c_time[0],
                       c_dwell[-1] + c_time[-1] - c_time[0]):
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
        logger.debug("{}: Measurement {}".format(_fname, measurement))
        for code in range(start, c_time.size):
            logger.debug("{}: Checking code {}".format(_fname, code))
            if has_overlap(time[measurement], dwell[measurement], c_time[code],
                           c_dwell[code]):
                # Record the intersection
                t1, d1 = get_overlap(time[measurement], dwell[measurement],
                                     c_time[code], c_dwell[code])
                if d1 > 0:
                    logger.debug("{}: Overlap found: {}, {}".format(
                        _fname, t1, d1))
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
    return (theta[positions], v[positions], h[positions], times1, dwells1,
            bundles)


def monotonic(x):
    """Check whether x is monomtically increasing."""
    dx = np.diff(x)
    return np.all(dx >= 0)


def has_overlap(x0, xd, y0, yd):
    """Return True if the ranges overlap.

    Parameters
    ----------
    x0, y0 : float
        The min values of the ranges
    xd, yd : float
        The widths of the ranges

    """
    return x0 + xd >= y0 and y0 + yd >= x0


def get_overlap(x0, xd, y0, yd):
    """Return the min edge and width of overlap.

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
