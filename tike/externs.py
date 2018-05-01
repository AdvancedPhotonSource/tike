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

import os.path
import ctypes
import glob
from . import utils
import logging


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['c_shared_lib',
           'c_art',
           'c_project',
           'c_coverage']


logger = logging.getLogger(__name__)

# Import shared library.
LIBTIKE = ctypes.CDLL('./tike/libtike.cpython-35m-x86_64-linux-gnu.so')


def c_art(data, x, y, theta, recon):
    LIBTIKE.art.restype = utils.as_c_void_p()
    return LIBTIKE.art(
            utils.as_c_float_p(data),
            utils.as_c_float_p(x),
            utils.as_c_float_p(y),
            utils.as_c_float_p(theta),
            utils.as_c_float_p(recon))


def c_project(obj, oxmin, oymin, ozmin, ox, oy, oz, theta, h, v, dsize, data):
    LIBTIKE.project.restype = utils.as_c_void_p()
    return LIBTIKE.project(
            utils.as_c_float_p(obj),
            utils.as_c_float(oxmin),
            utils.as_c_float(oymin),
            utils.as_c_float(ozmin),
            utils.as_c_int(ox),
            utils.as_c_int(oy),
            utils.as_c_int(oz),
            utils.as_c_float_p(theta),
            utils.as_c_float_p(h),
            utils.as_c_float_p(v),
            utils.as_c_int(dsize),
            utils.as_c_float_p(data))


def c_coverage(oxmin, oymin, ozmin, ox, oy, oz, theta, h, v, w, dsize, cov):
    LIBTIKE.coverage.restype = utils.as_c_void_p()
    return LIBTIKE.coverage(
            utils.as_c_float(oxmin),
            utils.as_c_float(oymin),
            utils.as_c_float(ozmin),
            utils.as_c_int(ox),
            utils.as_c_int(oy),
            utils.as_c_int(oz),
            utils.as_c_float_p(theta),
            utils.as_c_float_p(h),
            utils.as_c_float_p(v),
            utils.as_c_float_p(w),
            utils.as_c_int(dsize),
            utils.as_c_float_p(cov))
