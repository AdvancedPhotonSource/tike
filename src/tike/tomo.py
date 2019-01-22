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

"""Define functions for solving the tomography problem.

Coordinate Systems
==================
`theta, v, h`. `v, h` are the horizontal vertical directions perpendicular
to the probe direction where positive directions are to the right and up.
`theta` is the rotation angle around the vertical reconstruction
space axis, `z`. `z` is parallel to `v`, and uses the right hand rule to
determine reconstruction space coordinates `z, x, y`. `theta` is measured
from the `x` axis, so when `theta = 0`, `h` is parallel to `y`.

Functions
=========
Each public function in this module should have the following interface:

Parameters
----------
obj : (Z, X, Y, P) :py:class:`numpy.array` float
    An array of material properties. The first three dimensions `Z, X, Y`
    are spatial dimensions. The fourth dimension, `P`,  holds properties at
    each grid position: refractive indices, attenuation coefficents, etc.

        * (..., 0) : delta, the real phase velocity, the decrement of the
            refractive index.
        * (..., 1) : beta, the imaginary amplitude extinction / absorption
            coefficient.

obj_corner : (3, ) float
    The min corner (z, x, y) of the `obj`.
line_integrals : (M, V, H, P) :py:class:`numpy.array` float
    Integrals across the `obj` for each of the `probe` rays and
    P parameters.
probe : (V, H, P) :py:class:`numpy.array` float
    The initial parameters of the probe to be projected across
    the `obj`. The grid of each probe is `H` rays wide (the
    horizontal direction) and `V` rays tall (the vertical direction). The
    fourth dimension, `P`, holds parameters at each grid position:
    real and imaginary wave components
theta, v, h : (M, ) :py:class:`numpy.array` float
    The min corner (theta, v, h) of the `probe` for each measurement.
kwargs
    Keyword arguments specific to this function. `**kwargs` should always be
    included so that extra parameters are ignored instead of raising an error.

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tomopy
import numpy as np
from . import utils
from libtike import LIBTIKE
import logging

__author__ = "Doga Gursoy, Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ["reconstruct",
           "forward",
           ]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reconstruct(
        obj,
        theta,
        line_integrals,
        **kwargs
):
    """Reconstruct the `obj` using the given `algorithm`.

    Parameters
    ----------
    obj : (Z, X, Y, P) :py:class:`numpy.array` float
        The initial guess for the reconstruction.

    Returns
    -------
    obj : (Z, X, Y, P) :py:class:`numpy.array` float
        The updated obj grid.

    """
    lr = tomopy.recon(tomo=line_integrals.real,
                      theta=theta,
                      init_recon=obj.real,
                      **kwargs,
                      )
    li = tomopy.recon(tomo=line_integrals.imag,
                      theta=theta,
                      init_recon=obj.imag,
                      **kwargs,
                      )
    recon = np.empty(lr.shape, dtype=complex)
    recon.real = lr
    recon.imag = li
    return recon


def forward(
        obj,
        theta,
        **kwargs
):
    """Compute line integrals over an obj."""
    lr = tomopy.project(obj=obj.real, theta=theta, pad=False)
    li = tomopy.project(obj=obj.imag, theta=theta, pad=False)
    line_integrals = np.empty(lr.shape, dtype=complex)
    line_integrals.real = lr
    line_integrals.imag = li
    return line_integrals
