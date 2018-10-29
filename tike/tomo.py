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

obj_min : (3, ) float
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
                    probe, theta, v, h,
                    **kwargs):
    """Define an interface all functions in this module match.

    This function also sets default values for functions in this module.
    """
    if obj is None:
        # An inital guess is required
        raise ValueError()
    # obj = utils.as_float32(obj)  # complex data
    if obj_min is None:
        # The default origin is at the center of the object
        obj_min = - np.array(obj.shape) / 2  # (z, x, y)
    obj_min = utils.as_float32(obj_min)
    if probe is None:
        # Assume a full field geometry
        probe = np.ones([obj.shape[0], obj.shape[2]])
    # probe = utils.as_float32(probe)  # complex data
    if theta is None:
        # Angle definitions are required
        raise ValueError()
    theta = utils.as_float32(theta)
    if v is None:
        v = np.full(theta.shape, obj_min[0])
    v = utils.as_float32(v)
    if h is None:
        h = np.full(theta.shape, obj_min[2])
    h = utils.as_float32(h)
    assert theta.size == v.size == h.size, \
        "The size of theta, v, h must be the same as the number of probes."
    # Generate a grid of offset vectors
    V, H = probe.shape
    gv = (np.arange(V) + 0.5)
    gh = (np.arange(H) + 0.5)
    dv, dh = np.meshgrid(gv, gh, indexing='ij')
    # Duplicate the trajectory by size of probe
    M = theta.size
    th1 = np.repeat(theta, V*H).reshape(M, V, H)
    v1 = (np.repeat(v, V*H).reshape(M, V, H) + dv)
    h1 = (np.repeat(h, V*H).reshape(M, V, H) + dh)
    assert th1.shape == v1.shape == h1.shape
    # logger.info(" _tomo_interface says {}".format("Hello, World!"))
    return (obj, obj_min, probe, th1, v1, h1)


def reconstruct(obj=None, obj_min=None,
                probe=None, theta=None, v=None, h=None,
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
    Lr = tomopy.recon(tomo=line_integrals.real,
                      theta=theta,
                      algorithm=algorithm,
                      init_recon=obj.real,
                      num_iter=niter, **kwargs,
                      )
    Li = tomopy.recon(tomo=line_integrals.imag,
                      theta=theta,
                      algorithm=algorithm,
                      init_recon=obj.imag,
                      num_iter=niter, **kwargs,
                      )
    recon = np.empty(Lr.shape, dtype=complex)
    recon.real = Lr
    recon.imag = Li
    return recon
    # obj, obj_min, probe, theta, v, h \
    #     = _tomo_interface(obj, obj_min, probe, theta, v, h)
    # assert niter >= 0, "Number of iterations should be >= 0"
    # # Send data to c function
    # logger.info("{} on {:,d} element grid for {:,d} iterations".format(
    #             algorithm, obj.size, niter))
    # ngrid = obj.shape
    # line_integrals = utils.as_float32(line_integrals)
    # theta = utils.as_float32(theta)
    # v = utils.as_float32(v)
    # h = utils.as_float32(h)
    # obj = utils.as_float32(obj)
    # # Add new tomography algorithms here
    # # TODO: The size of this function may be reduced further if all recon
    # #   clibs have a standard interface. Perhaps pass unique params to a
    # #   generic struct or array.
    # if algorithm is "art":
    #     LIBTIKE.art.restype = utils.as_c_void_p()
    #     LIBTIKE.art(
    #         utils.as_c_float(obj_min[0]),
    #         utils.as_c_float(obj_min[1]),
    #         utils.as_c_float(obj_min[2]),
    #         utils.as_c_int(ngrid[0]),
    #         utils.as_c_int(ngrid[1]),
    #         utils.as_c_int(ngrid[2]),
    #         utils.as_c_float_p(line_integrals),
    #         utils.as_c_float_p(theta),
    #         utils.as_c_float_p(h),
    #         utils.as_c_float_p(v),
    #         utils.as_c_int(line_integrals.size),
    #         utils.as_c_float_p(obj),
    #         utils.as_c_int(niter))
    # elif algorithm is "sirt":
    #     LIBTIKE.sirt.restype = utils.as_c_void_p()
    #     LIBTIKE.sirt(
    #         utils.as_c_float(obj_min[0]),
    #         utils.as_c_float(obj_min[1]),
    #         utils.as_c_float(obj_min[2]),
    #         utils.as_c_int(ngrid[0]),
    #         utils.as_c_int(ngrid[1]),
    #         utils.as_c_int(ngrid[2]),
    #         utils.as_c_float_p(line_integrals),
    #         utils.as_c_float_p(theta),
    #         utils.as_c_float_p(h),
    #         utils.as_c_float_p(v),
    #         utils.as_c_int(line_integrals.size),
    #         utils.as_c_float_p(obj),
    #         utils.as_c_int(niter))
    # else:
    #     raise ValueError("The {} algorithm is not an available.".format(
    #         algorithm))
    # return obj


def forward(obj=None, obj_min=None,
            probe=None, theta=None, v=None, h=None,
            **kwargs):
    """Compute line integrals over an obj; i.e. simulate data acquisition."""
    Lr = tomopy.project(obj=obj.real, theta=theta, pad=False)
    Li = tomopy.project(obj=obj.imag, theta=theta, pad=False)
    line_integrals = np.empty(Lr.shape, dtype=complex)
    line_integrals.real = Lr
    line_integrals.imag = Li
    return line_integrals
    # obj, obj_min, probe, theta, v, h \
    #     = _tomo_interface(obj, obj_min, probe, theta, v, h)
    # logger.info("forward {:,d} element grid".format(obj.size))
    # logger.info("forward {:,d} rays".format(h.size))
    # ngrid = obj.shape
    # theta = utils.as_float32(theta)
    # v = utils.as_float32(v)
    # h = utils.as_float32(h)
    # line_integrals = np.zeros([*theta.shape, 2], dtype=float)
    # obj = obj.view(float).reshape(*obj.shape, 2)
    # # Send data to c function
    # for i in range(2):
    #     line = utils.as_float32(line_integrals[..., i])
    #     objt = utils.as_float32(obj[..., i])
    #     LIBTIKE.forward_project.restype = utils.as_c_void_p()
    #     LIBTIKE.forward_project(
    #         utils.as_c_float_p(objt),
    #         utils.as_c_float(obj_min[0]),
    #         utils.as_c_float(obj_min[1]),
    #         utils.as_c_float(obj_min[2]),
    #         utils.as_c_int(ngrid[0]),
    #         utils.as_c_int(ngrid[1]),
    #         utils.as_c_int(ngrid[2]),
    #         utils.as_c_float_p(theta),
    #         utils.as_c_float_p(v),
    #         utils.as_c_float_p(h),
    #         utils.as_c_int(theta.size),
    #         utils.as_c_float_p(line))
    #     line_integrals[..., i] = line
    # return line_integrals.view(complex)[..., 0]
