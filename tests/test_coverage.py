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
from numpy.testing import assert_allclose, assert_raises, assert_equal
from tike.scan import *
import matplotlib.pyplot as plt

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


# Test line_offsets

def test_probe_equal_to_pixel():
    p = Probe(width=1, aspect=1)
    assert len(p.line_offsets(pixel_size=1)) == 16**2
    assert 1 / p.line_width == 16


def test_probe_larger_than_pixel():
    p = Probe(width=2, aspect=1)
    assert len(p.line_offsets(pixel_size=1)) == 32**2
    assert 1 / p.line_width == 16


def test_probe_half_of_pixel_size():
    p = Probe(width=1/2, aspect=1)
    assert len(p.line_offsets(pixel_size=1)) == 8**2
    assert 1 / p.line_width == 16


def test_probe_equal_to_default_line_size():
    p = Probe(width=1/16, aspect=1)
    assert len(p.line_offsets(pixel_size=1)) == 1
    assert 1 / p.line_width == 16


def test_probe_smaller_than_default_line_size():
    try:
        p = Probe(width=1/32, aspect=1)
    except NotImplementedError:
        pass


def test_discrete_trajectory():
    def stationary(t):
        """Probe is stationary at location h = 8, v = 8"""
        return [0, 8, 8]

    answer = discrete_trajectory(stationary, tmin=0, tmax=0.65, dx=0.1, dt=1)
    truth = ([[0, 8, 8]], [0.65], [0])
    assert_equal(answer, truth)


# Test Coverage

def show_coverage(cov_map):
    [i, j, k] = np.array(cov_map.shape)
    plt.figure()
    plt.title(k//2)
    plt.imshow(cov_map[:, :, k//2], interpolation=None,)
    plt.yticks(np.arange(i).astype(int))
    plt.xticks(np.arange(j).astype(int))
    plt.ylabel('x - axis')
    plt.xlabel('yh - axis')
    plt.colorbar()
    plt.grid()
    # plt.clim([0, 10])
    plt.figure()
    plt.title(j//2)
    plt.imshow(cov_map[:, j//2, :], interpolation=None,)
    plt.yticks(np.arange(i).astype(int))
    plt.xticks(np.arange(k).astype(int))
    plt.ylabel('x - axis')
    plt.xlabel('zv - axis')
    plt.colorbar()
    plt.grid()
    # plt.clim([0, 10])
    plt.figure()
    plt.title(i//2)
    plt.imshow(cov_map[i//2, :, :], interpolation=None,)
    plt.yticks(np.arange(j).astype(int))
    plt.xticks(np.arange(k).astype(int))
    plt.ylabel('yh - axis')
    plt.xlabel('zv - axis')
    plt.colorbar()
    plt.grid()
    # plt.clim([0, 10])


def init_coverage():
    """Return Probe of width 1/16"""
    p = Probe(width=1/16, aspect=1)
    region = np.array([[-8/16, 8/16], [-8/16, 8/16], [-8/16, 8/16]])
    # FIXME: Tests fail if region is adjusted to region below
    # region = np.array([[-8/16, 8/16], [-8/16, 3/16], [-4/16, 8/16]])
    pixel_size = 1/16
    return p, region, pixel_size


def stationary(t):
    """Probe is stationary at location h = 2 + 1/32, v = 1/32"""
    return [0*t, 2/16 + 1/32 + 0*t, 1/32 + 0*t]


def horizontal_move(t, h_speed=-2/320):
    """Probe moves horizontally at h_speed [cm/s]"""
    return [0*t, 1/32 + h_speed*t, 2/16 + 1/32 + 0*t]


def vertical_move(t, v_speed=2/320):
    """Probe moves vertically at v_speed [cm/s]"""
    return [0*t, 1/32 + 0*t, 1/32 + v_speed*t]


def theta_move(t, Hz=1):
    """Probe rotates at rate Hz [2 Pi radians / s]"""
    theta = 2 * np.pi * Hz * t
    return [theta, 0*t, 0*t]


def test_stationary_coverage():
    """A beam of magnitude 10 at (:, 10, 8)."""
    p, region, pixel_size = init_coverage()
    cov_map = p.coverage(trajectory=stationary, region=region,
                         pixel_size=pixel_size, tmin=0, tmax=10, dt=1)
    show_coverage(cov_map)
    key = cov_map[8, :, :]
    truth = np.zeros([16, 16])
    truth[10, 8] = 10
    plt.figure()
    plt.plot(key[:, 8], 'o')
    assert_equal(key, truth)


def test_stationary_coverage_crop():
    """A beam of magnitude 10 at (:, 10, 8)."""
    p, region, pixel_size = init_coverage()
    region = np.array([[-0/16, 8/16], [-0/16, 4/16], [-8/16, 8/16]])
    cov_map = p.coverage(trajectory=stationary, region=region,
                         pixel_size=pixel_size, tmin=0, tmax=10, dt=1)
    show_coverage(cov_map)
    key = cov_map[1, :, :]
    truth = np.zeros([4, 16])
    truth[2, 8] = 10
    plt.figure()
    plt.plot(key[:, 8], 'o')
    assert_equal(key, truth)


def test_horizontal_coverage():
    # NOTE: The forward edge of the smear will be slightly larger. The two
    # edges even out as the time step approaches zero.
    p, region, pixel_size = init_coverage()
    cov_map = p.coverage(trajectory=horizontal_move, region=region,
                         pixel_size=pixel_size, tmin=0, tmax=40, dt=1)
    show_coverage(cov_map)
    key = cov_map[10, :, :]
    truth = np.zeros([16, 16])
    truth[5:18, 10] = 10
    truth[(4, 8), 10] = 5
    plt.figure()
    plt.plot(key[:, 10], 'o')
    # print(key[8, :])
    # assert_equal(key, truth)
    assert key[8, 10] >= 5 and key[8, 10] < 6
    assert key[4, 10] > 4 and key[4, 10] <= 5
    assert np.all(key[5:8, 10] == 10)


def test_vertical_coverage():
    p, region, pixel_size = init_coverage()
    cov_map = p.coverage(trajectory=vertical_move, region=region,
                         pixel_size=pixel_size, tmin=0, tmax=40, dt=1)
    show_coverage(cov_map)
    key = cov_map[4, :, :]
    truth = np.zeros([16, 16])
    truth[8, 9:12] = 10
    truth[8, (8, 12)] = 5
    plt.figure()
    plt.plot(key[8, :], 'o')
    # print(key[8, :])
    # assert_array_equal(key, truth)
    assert key[8, 8] >= 5 and key[8, 8] < 6
    assert key[8, 12] > 4 and key[8, 12] <= 5
    assert np.all(key[8, 9:12] == 10)


def test_theta_coverage():
    p, region, pixel_size = init_coverage()
    cov_map = p.coverage(trajectory=theta_move, region=region,
                         pixel_size=pixel_size, tmin=0, tmax=1, dt=0.5)
    show_coverage(cov_map)


# def test_show_plots():
#     plt.show()
