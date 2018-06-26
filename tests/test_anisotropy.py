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
from tike.probe import *
import matplotlib.pyplot as plt

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


def init_coverage():
    """Return Probe of width 1/16"""
    p = Probe(width=1, aspect=1)
    region = np.array([[-0.5, 0.5], [-1.5, 1.5], [-1.5, 1.5]])
    # FIXME: Tests fail if region is adjusted to region below
    # region = np.array([[-8/16, 8/16], [-8/16, 3/16], [-4/16, 8/16]])
    pixel_size = 1.
    return p, region, pixel_size


def all_x(t):
    return np.pi + 0.*t, 1+0*t, 0*t


def all_y(t):
    return -np.pi/2 + 0.*t, 0*t, 0*t

def split_z(t):
    return 0*t, 0*t, 0*t

def round(t):
    return np.pi/3 + np.pi*t/10, 0*t, 0*t


def test_stationary_coverage_x():
    p, region, pixel_size = init_coverage()
    cov_map = p.coverage(trajectory=all_x, region=region,
                         pixel_size=pixel_size, tmin=0, tmax=10, tstep=1,
                         anisotropy=8)
    # cov_map = cov_map.reshape((3, 3, 4))
    cov_map = cov_map
    truth = np.zeros(cov_map.shape)
    truth[0, :, 0, 0] = 10
    assert_allclose(truth, cov_map, atol=1e-20)


def test_stationary_coverage_y():
    p, region, pixel_size = init_coverage()
    cov_map = p.coverage(trajectory=all_y, region=region,
                         pixel_size=pixel_size, tmin=0, tmax=10, tstep=1,
                         anisotropy=3)
    # cov_map = cov_map.reshape((3, 3, 4))
    cov_map = cov_map
    truth = np.zeros(cov_map.shape)
    truth[0, 1, :, 1] = 10
    assert_allclose(truth, cov_map, atol=1e-6)


def test_split_z():
    pixel_size = 1.
    p = Probe(width=1, aspect=1)
    region = np.array([[-2, 2], [-2, 2], [-2, 2]])
    cov_map = p.coverage(trajectory=split_z, region=region,
                         pixel_size=pixel_size, tmin=0, tmax=10, tstep=1,
                         anisotropy=False)
    truth = np.zeros(cov_map.shape)
    truth[1:3,:,1:3] = 2.5
    print()
    print(cov_map[0,...])
    print(cov_map[1,...])
    assert_allclose(truth, cov_map, atol=1e-20)


def test_random_equivalent():
    pixel_size = 1.
    p = Probe(width=1, aspect=1)
    region = np.array([[-1, 1], [-2, 2], [-2, 2]])
    ani_map = p.coverage(trajectory=round, region=region,
                         pixel_size=pixel_size, tmin=0, tmax=10, tstep=1,
                         anisotropy=7)
    ani_map = np.sum(ani_map, axis=3)
    cov_map = p.coverage(trajectory=round, region=region,
                         pixel_size=pixel_size, tmin=0, tmax=10, tstep=1,
                         anisotropy=False)
    assert_allclose(ani_map, cov_map, atol=1e-5)
