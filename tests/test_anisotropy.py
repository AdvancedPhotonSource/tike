#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2018, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from tike import *

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


def init_coverage():
    """Create a (1, 1) probe and a (1, 3, 3) grid centered on the origin."""
    probe_grid = np.ones((16, 16))
    probe_size = (1, 1)
    region = np.zeros((1, 3, 3, 8))
    region_corner = [-0.5, -1.5, -1.5]
    region_size = [1, 3, 3]
    return probe_grid, probe_size, region, region_corner, region_size


def test_stationary_coverage_x():
    def all_x(t):
        return np.pi + 0*t, 0*t - 0.5, 0*t + 0.5
    probe_grid, probe_size, \
        region, region_corner, region_size = init_coverage()
    region = np.zeros((1, 3, 3, 8))
    theta, v, h, dwell, times = discrete_trajectory(all_x,
                                                    tmin=0, tmax=10, tstep=1,
                                                    xstep=1/32)
    cov_map = coverage(region, region_corner, region_size,
                       probe_grid, probe_size, theta, v, h, dwell)
    truth = np.zeros(cov_map.shape)
    truth[0, :, 0, 0] = 10
    np.testing.assert_equal(truth, cov_map)


def test_stationary_coverage_y():
    def all_y(t):
        return -np.pi/2 + 0*t, 0*t - 0.5, 0*t - 0.5
    probe_grid, probe_size, \
        region, region_corner, region_size = init_coverage()
    region = np.zeros((1, 3, 3, 3))
    theta, v, h, dwell, times = discrete_trajectory(all_y,
                                                    tmin=0, tmax=10, tstep=1,
                                                    xstep=1/32)
    cov_map = coverage(region, region_corner, region_size,
                       probe_grid, probe_size, theta, v, h, dwell)
    # cov_map = cov_map.reshape((3, 3, 4))
    cov_map = cov_map
    truth = np.zeros(cov_map.shape)
    truth[0, 1, :, 1] = 10
    np.testing.assert_equal(truth, cov_map)


def test_split_z():
    """A probe can be split across z slices."""
    probe_grid = np.ones((16, 16))
    probe_size = (1, 1)
    region = np.zeros((4, 4, 4, 1))
    region_corner = [-2, -2, -2]
    region_size = [4, 4, 4]

    def split_z(t):
        return 0*t, 0*t - 0.5, 0*t - 0.5

    theta, v, h, dwell, times = discrete_trajectory(split_z,
                                                    tmin=0, tmax=10, tstep=1,
                                                    xstep=1/32)
    cov_map = coverage(region, region_corner, region_size,
                       probe_grid, probe_size, theta, v, h, dwell)[..., 0]
    truth = np.zeros(cov_map.shape)
    truth[1:3, :, 1:3] = 2.5
    np.testing.assert_equal(truth, cov_map,)


def test_Nbin_equivalent():
    """A coverage map with 1 or many angular bins has similar result."""
    # Define a trajectory for an origin-centered probe rotating once
    def round(t):
        return np.pi/3 + np.pi*t/10, 0*t - 0.5, 0*t - 0.5
    # Define the probe and grid extents
    probe_grid = np.ones((16, 16))
    probe_size = (1, 1)
    region_corner = [-1, -2, -2]
    region_size = [2, 4, 4]
    # Discretize the trajectory
    theta, v, h, dwell, times = discrete_trajectory(round,
                                                    tmin=0, tmax=10, tstep=1,
                                                    xstep=1/32)
    # Compute coverage for one and many bins
    region1 = np.zeros((2, 4, 4, 1))
    one_bin_map = coverage(region1, region_corner, region_size,
                           probe_grid, probe_size, theta, v, h, dwell)
    region7 = np.zeros((2, 4, 4, 7))
    any_bin_map = coverage(region7, region_corner, region_size,
                           probe_grid, probe_size, theta, v, h, dwell)
    np.testing.assert_allclose(np.sum(one_bin_map, axis=3),
                               np.sum(any_bin_map, axis=3), atol=1e-4)
