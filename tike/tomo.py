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
This module contains functions for solving the tomography problem.

Coordinate Systems
==================
`theta, h, v`. `h, v` are the horizontal vertical directions perpendicular
to the probe direction where positive directions are to the right and up
respectively. `theta` is the rotation angle around the vertical reconstruction
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

obj_min : (3, ) float
    The min corner (z, x, y) of the `obj`.
line_integrals : (M, H, V, P) :py:class:`numpy.array` float
    Integrals across the `obj` for each of the `probe` rays and
    P parameters.
probe : (H, V, P) :py:class:`numpy.array` float
    The initial parameters of the probe to be projected across
    the `obj`. The grid of each probe is `H` rays wide (the
    horizontal direction) and `V` rays tall (the vertical direction). The
    fourth dimension, `P`, holds parameters at each grid position:
    real and imaginary wave components
theta, h, v : (M, ) :py:class:`numpy.array` float
    The min corner (theta, h, v) of the `probe` for each measurement.
kwargs
    Keyword arguments specific to this function. `**kwargs` should always be
    included so that extra parameters are ignored instead of raising an error.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from . import utils
from tike.externs import LIBTIKE
import logging

__author__ = "Doga Gursoy, Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ["reconstruct",
           "forward",
           ]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _tomo_interface(obj, obj_min,
                    probe, theta, h, v,
                    **kwargs):
    """A function whose interface all functions in this module matches.

    This function also sets default values for functions in this module.
    """
    if obj is None:
        raise ValueError()
    obj = utils.as_float32(obj)
    if obj_min is None:
        obj_min = (-0.5, -0.5, -0.5)  # (z, x, y)
    obj_min = utils.as_float32(obj_min)
    if probe is None:
        raise ValueError()
    probe = utils.as_float32(probe)
    if theta is None:
        raise ValueError()
    theta = utils.as_float32(theta)
    if h is None:
        h = np.full(theta.shape, obj_min[2])
    h = utils.as_float32(h)
    if v is None:
        v = np.full(theta.shape, obj_min[0])
    v = utils.as_float32(v)
    assert theta.size == h.size == v.size, \
        "The size of theta, h, v must be the same as the number of probes."
    # Generate a grid of offset vectors
    H, V = probe.shape
    gh = (np.arange(H) + 0.5)
    gv = (np.arange(V) + 0.5)
    dh, dv = np.meshgrid(gh, gv, indexing='ij')
    # Duplicate the trajectory by size of probe
    M = theta.size
    th1 = np.repeat(theta, H*V).reshape(M, H, V)
    h1 = (np.repeat(h, H*V).reshape(M, H, V) + dh)
    v1 = (np.repeat(v, H*V).reshape(M, H, V) + dv)
    assert th1.shape == h1.shape == v1.shape
    # logger.info(" _tomo_interface says {}".format("Hello, World!"))
    return (obj, obj_min, probe, th1, h1, v1)


def reconstruct(obj=None, obj_min=None,
                probe=None, theta=None, h=None, v=None,
                line_integrals=None,
                algorithm=None, niter=0, **kwargs):
    """Reconstruct the `obj` using the given `algorithm`.

    Parameters
    ----------
    obj : (Z, X, Y, P) :py:class:`numpy.array` float
        The initial guess for the reconstruction.
    algorithm : string
        The name of one of the following algorithms to use for reconstructing:

            * art : Algebraic Reconstruction Technique
                :cite:`gordon1970algebraic`.
            * sirt : Simultaneous Iterative Reconstruction Technique.

    niter : int
        The number of iterations to perform

    Returns
    -------
    obj : (Z, X, Y, P) :py:class:`numpy.array` float
        The updated obj grid.
    """
    obj, obj_min, probe, theta, h, v \
        = _tomo_interface(obj, obj_min, probe, theta, h, v)
    assert niter >= 0, "Number of iterations should be >= 0"
    # Send data to c function
    logger.info("{} on {:,d} element grid for {:,d} iterations".format(
                algorithm, obj.size, niter))
    ngrid = obj.shape
    line_integrals = utils.as_float32(line_integrals)
    theta = utils.as_float32(theta)
    h = utils.as_float32(h)
    v = utils.as_float32(v)
    obj = utils.as_float32(obj)
    # Add new tomography algorithms here
    # TODO: The size of this function may be reduced further if all recon clibs
    #   have a standard interface. Perhaps pass unique params to a generic
    #   struct or array.
    if algorithm is "art":
        LIBTIKE.art.restype = utils.as_c_void_p()
        LIBTIKE.art(
            utils.as_c_float(obj_min[0]),
            utils.as_c_float(obj_min[1]),
            utils.as_c_float(obj_min[2]),
            utils.as_c_int(ngrid[0]),
            utils.as_c_int(ngrid[1]),
            utils.as_c_int(ngrid[2]),
            utils.as_c_float_p(line_integrals),
            utils.as_c_float_p(theta),
            utils.as_c_float_p(h),
            utils.as_c_float_p(v),
            utils.as_c_int(line_integrals.size),
            utils.as_c_float_p(obj),
            utils.as_c_int(niter))
    elif algorithm is "sirt":
        LIBTIKE.sirt.restype = utils.as_c_void_p()
        LIBTIKE.sirt(
            utils.as_c_float(obj_min[0]),
            utils.as_c_float(obj_min[1]),
            utils.as_c_float(obj_min[2]),
            utils.as_c_int(ngrid[0]),
            utils.as_c_int(ngrid[1]),
            utils.as_c_int(ngrid[2]),
            utils.as_c_float_p(line_integrals),
            utils.as_c_float_p(theta),
            utils.as_c_float_p(h),
            utils.as_c_float_p(v),
            utils.as_c_int(line_integrals.size),
            utils.as_c_float_p(obj),
            utils.as_c_int(niter))
    else:
        raise ValueError("The {} algorithm is not an available.".format(
            algorithm))
    return obj


def forward(obj=None, obj_min=None,
            probe=None, theta=None, h=None, v=None,
            **kwargs):
    """Compute line integrals over an obj; i.e. simulate data acquisition.
    """
    obj, obj_min, probe, theta, h, v \
        = _tomo_interface(obj, obj_min, probe, theta, h, v)
    # TODO: Remove zero valued probe rays
    th1 = theta
    h1 = h
    v1 = v
    line_integrals = np.zeros(th1.shape, dtype=np.float32)
    # Send data to c function
    logger.info("forward {:,d} element grid".format(obj.size))
    logger.info("forward {:,d} rays".format(line_integrals.size))
    obj = utils.as_float32(obj)
    ngrid = obj.shape
    th1 = utils.as_float32(th1)
    h1 = utils.as_float32(h1)
    v1 = utils.as_float32(v1)
    line_integrals = utils.as_float32(line_integrals)
    LIBTIKE.forward_project.restype = utils.as_c_void_p()
    LIBTIKE.forward_project(
        utils.as_c_float_p(obj),
        utils.as_c_float(obj_min[0]),
        utils.as_c_float(obj_min[1]),
        utils.as_c_float(obj_min[2]),
        utils.as_c_int(ngrid[0]),
        utils.as_c_int(ngrid[1]),
        utils.as_c_int(ngrid[2]),
        utils.as_c_float_p(th1),
        utils.as_c_float_p(h1),
        utils.as_c_float_p(v1),
        utils.as_c_int(th1.size),
        utils.as_c_float_p(line_integrals))
    return line_integrals
