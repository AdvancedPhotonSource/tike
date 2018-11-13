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
from tike.tomo import *

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


# def test_forward_project_hv_quadrants1():
#     gmin = [0, 0, 0]
#     obj = np.zeros((2, 3, 2), dtype=complex)
#     obj[0, 1, 0] = 1
#     theta, h, v = [0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1]
#     pgrid = np.ones((1, 1))
#     integral = forward(obj, gmin,
#                        pgrid, theta, v, h)
#     truth = np.array([1, 0, 0, 0]).reshape(4, 1, 1)
#     np.testing.assert_equal(truth, integral)
#
#
# def test_forward_project_hv_quadrants2():
#     gsize = np.array([3, 2, 2])
#     gmin = -gsize / 2.0
#     obj = np.zeros((3, 2, 2), dtype=complex)
#     obj[2, 0, 1] = 1
#     theta, h, v = [0, 0], [-1, 0], [0.5, 0.5]
#     pgrid = np.ones((1, 1))
#     integral = forward(obj, gmin,
#                        pgrid, theta, v, h)
#     truth = np.array([0, 1]).reshape(2, 1, 1)
#     np.testing.assert_equal(truth, integral)
#
#
# def test_forward_project_hv_quadrants4():
#     gsize = np.array([1, 2, 2])
#     gmin = -gsize / 2.0
#     obj = np.zeros((1, 2, 2), dtype=complex)
#     obj[0, :, 1] = 1
#     theta, h, v = [0], [-0.5], [-0.5]
#     pgrid = np.ones((1, 1))
#     psize = (1, 1)
#     integral = forward(obj, gmin,
#                        pgrid, theta, v, h)
#     truth = np.array([2]).reshape(1, 1, 1)
#     np.testing.assert_equal(truth, integral)
