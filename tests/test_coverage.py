#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2016, UChicago Argonne, LLC. All rights reserved.         #
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
import matplotlib.pyplot as plt
from tike.coverage import *
from tike.trajectory import discrete_trajectory

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

pgrid = np.ones((1, 1))
psize = (1, 1)


def test_coverage_hv_quadrants1_crop():
    theta, h, v = [0], [-0.5], [0]
    gmin = np.array([-2, 0, -1])
    gsize = np.array([5, 1, 3])
    grid = np.zeros((5, 1, 3, 1))
    cov_map = coverage(grid, gmin, gsize,
                       pgrid, psize,
                       theta, v, h)[:, 0, :, 0]
    truth = np.array([[0,  0,  0,  0,  0],
                      [0,  0,  1,  0,  0],
                      [0,  0,  0,  0,  0]])
    truth = np.moveaxis(truth, 1, 0)
    np.testing.assert_equal(truth, cov_map)


def test_coverage_hv_quadrants1():
    gsize = np.array([1, 2, 2])
    gmin = -gsize / 2.0
    grid = np.zeros([1, 2, 2, 1])
    theta, h, v = [0], [0], [-0.5]
    cov_map = coverage(grid, gmin, gsize,
                       pgrid, psize,
                       theta, v, h)[0, ..., 0]
    truth = np.array([[0.,  1.],
                      [0.,  1.]])
    np.testing.assert_equal(truth, cov_map)


def test_coverage_hv_quadrants1b():
    grid = np.zeros([1, 2, 2, 1])
    gsize = np.array([1, 2, 2])
    gmin = -gsize / 2.0
    theta, h, v = [np.pi/2], [0.], [-0.5]
    cov_map = coverage(grid, gmin, gsize,
                       pgrid, psize,
                       theta, v, h)[0, ..., 0]
    truth = np.array([[1.,  1.],
                      [0.,  0.]])
    np.testing.assert_equal(truth, cov_map)


def test_coverage_hv_quadrants1c():
    grid = np.zeros([1, 2, 2, 1])
    gsize = np.array([1, 2, 2])
    gmin = -gsize / 2.0
    theta, h, v = [-np.pi/2], [0], [-0.5]
    cov_map = coverage(grid, gmin, gsize,
                       pgrid, psize,
                       theta, v, h)[0, ..., 0]
    truth = np.array([[0.,  0.],
                      [1.,  1.]])
    np.testing.assert_equal(truth, cov_map)


def test_coverage_hv_quadrants1d():
    grid = np.zeros([1, 2, 2, 1])
    gsize = np.array([1, 2, 2])
    gmin = -gsize / 2.0
    theta, h, v = [np.pi], [0.], [-0.5]
    cov_map = coverage(grid, gmin, gsize,
                       pgrid, psize,
                       theta, v, h)[0, ..., 0]
    truth = np.array([[1.,  0.],
                      [1.,  0.]])
    np.testing.assert_equal(truth, cov_map)


def test_coverage_hv_quadrants2():
    grid = np.zeros([2, 2, 2, 1])
    gsize = np.array([2, 2, 2])
    gmin = -gsize / 2.0
    obj = np.zeros((2, 2, 2))
    theta, h, v = [0], [-1], [0]
    cov_map = coverage(grid, gmin, gsize,
                       pgrid, psize,
                       theta, v, h)[..., 0]
    obj[1, :, 0] = 1
    np.testing.assert_equal(obj, cov_map)


def test_coverage_hv_quadrants4():
    grid = np.zeros([2, 2, 2, 1])
    gsize = np.array([2, 2, 2])
    gmin = -gsize / 2.0
    obj = np.zeros((2, 2, 2))
    theta, h, v = [0], [0], [-1]
    cov_map = coverage(grid, gmin, gsize,
                       pgrid, psize,
                       theta, v, h)[..., 0]
    obj[0, :, 1] = 1
    np.testing.assert_equal(obj, cov_map)


def show_coverage(cov_map):
    [i, j, k] = np.array(cov_map.shape)
    plt.figure()
    plt.title(k//2)
    plt.imshow(cov_map[:, :, k//2], interpolation=None,)
    plt.yticks(np.arange(i).astype(int))
    plt.xticks(np.arange(j).astype(int))
    plt.ylabel('zv - axis')
    plt.xlabel('x - axis')
    plt.colorbar()
    plt.grid()
    # plt.clim([0, 10])
    plt.figure()
    plt.title(j//2)
    plt.imshow(cov_map[:, j//2, :], interpolation=None,)
    plt.yticks(np.arange(i).astype(int))
    plt.xticks(np.arange(k).astype(int))
    plt.ylabel('z - axis')
    plt.xlabel('yh - axis')
    plt.colorbar()
    plt.grid()
    # plt.clim([0, 10])
    plt.figure()
    plt.title(i//2)
    plt.imshow(cov_map[i//2, :, :], interpolation=None,)
    plt.yticks(np.arange(j).astype(int))
    plt.xticks(np.arange(k).astype(int))
    plt.ylabel('x - axis')
    plt.xlabel('yh - axis')
    plt.colorbar()
    plt.grid()
    # plt.clim([0, 10])


def init_coverage():
    """Return Probe of width 1/16"""
    probe_grid = np.ones([4, 2])
    probe_size = [1/16, 1/16]
    region = np.zeros([16, 16, 16, 1])
    region_corner = [-8/16, -8/16, -8/16]
    region_size = [1, 1, 1]
    # FIXME: Tests fail if region is adjusted to region below
    # region = np.array([[-8/16, 8/16], [-8/16, 3/16], [-4/16, 8/16]])
    return probe_grid, probe_size, region, region_corner, region_size


def stationary(t):
    """Probe is stationary at location h = 2, v = 0"""
    return 0.*t, 0*t, 2/16 + 0*t


def test_stationary_coverage():
    """A beam of magnitude 10 at (:, 10, 8)."""
    probe_grid, probe_size, \
        region, region_corner, region_size = init_coverage()
    theta, v, h, dwell, times = discrete_trajectory(stationary,
                                                    tmin=0, tmax=10, tstep=1,
                                                    xstep=1/32)
    cov_map = coverage(region, region_corner, region_size,
                       probe_grid, probe_size, theta, v, h, dwell)[..., 0]

    # show_coverage(cov_map)
    key = cov_map[:, 8, :]
    truth = np.zeros([16, 16])
    truth[8, 10] = 10
    # plt.figure()
    # plt.plot(key[8, :], 'o')
    np.testing.assert_equal(key, truth)


def test_stationary_coverage_crop():
    """A beam of magnitude 10 at (:, 10, 8)."""
    probe_grid, probe_size, \
        region, region_corner, region_size = init_coverage()
    theta, v, h, dwell, times = discrete_trajectory(stationary,
                                                    tmin=0, tmax=10, tstep=1,
                                                    xstep=1/32)
    region = np.zeros([16, 8, 4, 1])
    region_corner = [-8/16, -0/16, -0/16]
    region_size = [1, 8/16, 4/16]
    cov_map = coverage(region, region_corner, region_size,
                       probe_grid, probe_size, theta, v, h, dwell)[..., 0]

    # show_coverage(cov_map)
    key = cov_map[:, 1, :]
    truth = np.zeros([16, 4])
    truth[8, 2] = 10
    # plt.figure()
    # plt.plot(key[8, :], 'o')
    np.testing.assert_equal(key, truth)


def horizontal_move(t, h_speed=-2/320):
    """Probe moves horizontally at h_speed [cm/s]"""
    return 0.*t, 2/16 + 0*t, h_speed*t


def test_horizontal_coverage():
    # NOTE: The forward edge of the smear will be slightly larger. The two
    # edges even out as the time step approaches zero.
    probe_grid, probe_size, \
        region, region_corner, region_size = init_coverage()
    theta, v, h, dwell, times = discrete_trajectory(horizontal_move,
                                                    tmin=0, tmax=40, tstep=1,
                                                    xstep=1/32)
    cov_map = coverage(region, region_corner, region_size,
                       probe_grid, probe_size, theta, v, h, dwell)[..., 0]
    # show_coverage(cov_map)
    key = cov_map[:, 10, :]
    truth = np.zeros([16, 16])
    truth[10, 5:18] = 10
    truth[10, (4, 8)] = 5
    # plt.figure()
    # plt.plot(key[10, :], 'o')
    print(key[10, :])
    # np.testing.assert_equal(key, truth)
    assert key[10, 8] >= 5 and key[10, 8] < 6
    assert key[10, 4] > 4 and key[10, 4] <= 5
    assert np.all(key[10, 5:8] == 10)


def vertical_move(t, v_speed=2/320):
    """Probe moves vertically at v_speed [cm/s]"""
    return 0.*t, v_speed*t, 0*t


def test_vertical_coverage():
    probe_grid, probe_size, \
        region, region_corner, region_size = init_coverage()
    theta, v, h, dwell, times = discrete_trajectory(vertical_move,
                                                    tmin=0, tmax=40, tstep=1,
                                                    xstep=1/32)
    cov_map = coverage(region, region_corner, region_size,
                       probe_grid, probe_size, theta, v, h, dwell)[..., 0]
    # show_coverage(cov_map)
    key = cov_map[:, 4, :]
    truth = np.zeros([16, 16])
    truth[9:12, 8] = 10
    truth[(8, 12), 8] = 5
    # plt.figure()
    # plt.plot(key[:, 8], 'o')
    # print(key[8, :])
    # assert_array_equal(key, truth)
    assert key[8, 8] >= 5 and key[8, 8] < 6
    assert key[12, 8] > 4 and key[12, 8] <= 5
    assert np.all(key[9:12, 8] == 10)


def theta_move(t, Hz=1):
    """Probe rotates at rate Hz [2 Pi radians / s]"""
    theta = 2 * np.pi * Hz * t
    return theta, 0*t, 0*t - 1/16/2


def test_theta_coverage():
    probe_grid, probe_size, \
        region, region_corner, region_size = init_coverage()
    theta, v, h, dwell, times = discrete_trajectory(theta_move,
                                                    tmin=0, tmax=1, tstep=0.5,
                                                    xstep=1/32)
    region = np.zeros([4, 16, 16, 1])
    region_corner = [-2/16, -8/16, -8/16]
    region_size = [4/16, 16/16, 16/16]
    cov_map = coverage(region, region_corner, region_size,
                       probe_grid, probe_size, theta, v, h, dwell)[..., 0]
    # np.save('tests/theta_coverage.npy', cov_map)
    truth = np.load('tests/theta_coverage.npy')
    # show_coverage(cov_map)
    print("Computed map\n{}\n".format(cov_map))
    print("True map\n{}\n".format(truth))
    # np.testing.assert_equal(cov_map, truth)


if __name__ == '__main__':
    test_stationary_coverage()
    test_horizontal_coverage()
    test_vertical_coverage()
    test_theta_coverage()
    plt.show()
