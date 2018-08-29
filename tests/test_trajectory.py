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
from tike.trajectory import *

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


def test_discrete_trajectory():
    def stationary(t):
        """Probe is stationary at location h = 8, v = 8"""
        return 0*t, 8 + 0*t, 8 + 0*t

    answer = discrete_trajectory(stationary,
                                 tmin=0, tmax=0.65, xstep=0.1, tstep=1)
    truth = ([0], [8], [8], [0.65], [0])
    np.testing.assert_equal(answer, truth)


def test_coded_exposure():
    c_time = np.arange(11)
    c_dwell = np.ones(11) * 0.5

    time = np.array([-1., 0.8, 1.8, 3.0, 4.1, 4.2, 6.1,
                     7.5, 8.6, 8.9, 8.9, 8.9, 20, 21])
    dwell = np.array([0.1, 0.2, 0.4, 0.5, 0.1, 0.1, 0.6,
                      0.2, 0.2,   2,   0, 0.3, 1.0, 1.0])

    theta = np.arange(time.size)
    h = np.arange(time.size)
    v = np.arange(time.size)

    th1, h1, v1, t1, d1, b1 = coded_exposure(theta, h, v, time, dwell, c_time,
                                             c_dwell)

    np.testing.assert_equal(th1, [2, 3, 4, 5, 6, 9, 11, 9])
    np.testing.assert_equal(h1, [2, 3, 4, 5, 6, 9, 11, 9])
    np.testing.assert_equal(v1, [2, 3, 4, 5, 6, 9, 11, 9])
    np.testing.assert_equal(t1, [2., 3., 4.1, 4.2, 6.1, 9., 9., 10.])
    np.testing.assert_allclose(d1, [0.2, 0.5, 0.1, 0.1, 0.4, 0.5, 0.2, 0.5])
    np.testing.assert_equal(b1, [0, 1, 2, 4, 5, 7])


if __name__ == '__main__':
    test_discrete_trajectory()
    test_coded_exposure()
