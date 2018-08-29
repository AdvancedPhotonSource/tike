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
`theta, h, v`. `h, v` are the
horizontal vertical directions perpendicular to the probe direction
where positive directions are to the right and up respectively. `theta` is
the rotation angle around the vertical reconstruction space axis, `z`. `z`
is parallel to `v`, and uses the right hand rule to determine
reconstruction space coordinates `z, x, y`. `theta` is measured from the
`x` axis, so when `theta = 0`, `h` is parallel to `y`.

Functions
=========
Each public function in this module should have the following interface:

Parameters
----------
object_grid : (Z, X, Y, P) :py:class:`numpy.array` float
    An array of material properties. The first three dimensions `Z, X, Y`
    are spatial dimensions. The fourth dimension, `P`,  holds properties at
    each grid position: refractive indices, attenuation coefficents, etc.
object_min : (3, ) float
    The min corner (z, x, y) of `object_grid`.
object_size : (3, ) float
    The side lengths (z, x, y) of `object_grid` along each dimension.
probe_grid : (M, H, V, P) :py:class:`numpy.array` float
    The parameters of the `M` probes to be collected or projected across
    `object_grid`. The grid of each probe is `H` rays wide (the
    horizontal direction) and `V` rays tall (the vertical direction). The
    fourth dimension, `P`, holds parameters at each grid position:
    measured intensity, relative phase shift, etc.
probe_size : (2, ) float
    The side lengths (h, v) of `probe_grid` along each dimension. The
    dimensions of each slice of probe_grid is the same for simplicity.
theta, h, v : (M, ) :py:class:`numpy.array` float
    The min corner (theta, h, v) of each `M` slice of `probe_grid`.
kwargs
    Keyword arguments specific to this function. `**kwargs` should always be
    included so that extra parameters are ignored instead of raising an error.

Returns
-------
output : :py:class:`numpy.array`
    Output specific to this function matching conventions for input
    parameters.
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
           "backward",
           ]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _tomo_interface(object_grid, object_min, object_size,
                    probe_grid, probe_size, theta, h, v,
                    **kwargs):
    """A function whose interface all functions in this module matches.

    This function also sets default values for functions in this module.
    """
    if object_grid is None:
        raise ValueError()
    object_grid = utils.as_float32(object_grid)
    if object_min is None:
        object_min = (-0.5, -0.5, -0.5)
    object_min = utils.as_float32(object_min)
    if object_size is None:
        object_size = (1.0, 1.0, 1.0)
    object_size = utils.as_float32(object_size)
    if probe_grid is None:
        raise ValueError()
    probe_grid = utils.as_float32(probe_grid)
    if probe_size is None:
        probe_size = (1, 1)
    probe_size = utils.as_float32(probe_size)
    if theta is None:
        raise ValueError()
    theta = utils.as_float32(theta)
    if h is None:
        h = np.full(theta.shape, -0.5)
    h = utils.as_float32(h)
    if v is None:
        v = np.full(theta.shape, -0.5)
    v = utils.as_float32(v)
    assert np.all(object_size > 0), "Object dimensions must be > 0."
    assert np.all(probe_size > 0), "Probe dimensions must be > 0."
    assert theta.size == h.size == v.size == probe_grid.shape[0], \
        "The size of theta, h, v must be the same as the number of probes."
    # Duplicate the trajectory by size of probe_grid
    M, H, V, = probe_grid.shape
    dh, dv = line_offsets(H, V, probe_size)
    th1 = np.repeat(theta, H*V).reshape(M, H, V)
    h1 = (np.repeat(h, H*V).reshape(M, H, V) + dh)
    v1 = (np.repeat(v, H*V).reshape(M, H, V) + dv)
    assert th1.shape == h1.shape == v1.shape == probe_grid.shape
    # logger.info(" _tomo_interface says {}".format("Hello, World!"))
    return (object_grid, object_min, object_size,
            probe_grid, probe_size, th1, h1, v1)


def line_offsets(H, V, probe_size):
    """Generate h, v line offsets from the min corner.

    Returns
    -------
    dh, dv : (H, V) :py:class:`numpy.array` float [cm]
        The offsets in the horizontal and vertical directions
    """
    # Generate a grid of offset vectors
    gh = (np.linspace(0, probe_size[0], H, endpoint=False)
          + probe_size[0] / H / 2)
    gv = (np.linspace(0, probe_size[1], V, endpoint=False)
          + probe_size[1] / V / 2)
    dh, dv = np.meshgrid(gh, gv, indexing='ij')
    assert dv.shape == dh.shape == (H, V)
    return dh, dv


def reconstruct(object_grid=None, object_min=None, object_size=None,
                probe_grid=None, probe_size=None,
                theta=None, h=None, v=None,
                algorithm=None, **kwargs):
    """Reconstruct the `object_grid` using the given `algorithm`.

    Parameters
    ----------
    object_grid : (Z, X, Y, P) :py:class:`numpy.array` float
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
    object_grid : (Z, X, Y, P) :py:class:`numpy.array` float
        The updated object grid.
    """
    object_grid, object_min, object_size, probe_grid, probe_size, theta, h, v \
        = _tomo_interface(object_grid, object_min, object_size,
                          probe_grid, probe_size, theta, h, v)
    assert niter >= 0, "Number of iterations should be >= 0"
    # Send data to c function
    logger.info("{} on {:,d} element grid for {:,d} iterations".format(
                algorithm, object_grid.size, niter))
    ngrid = object_grid.shape
    probe_grid = utils.as_float32(probe_grid)
    theta = utils.as_float32(theta)
    h = utils.as_float32(h)
    v = utils.as_float32(v)
    object_grid = utils.as_float32(object_grid)
    # Add new tomography algorithms here
    # TODO: The size of this function may be reduced further if all recon clibs
    #   have a standard interface. Perhaps pass unique params to a generic
    #   struct or array.
    if algorithm is "art":
        LIBTIKE.art.restype = utils.as_c_void_p()
        LIBTIKE.art(
            utils.as_c_float(object_min[0]),
            utils.as_c_float(object_min[1]),
            utils.as_c_float(object_min[2]),
            utils.as_c_float(object_size[0]),
            utils.as_c_float(object_size[1]),
            utils.as_c_float(object_size[2]),
            utils.as_c_int(ngrid[0]),
            utils.as_c_int(ngrid[1]),
            utils.as_c_int(ngrid[2]),
            utils.as_c_float_p(probe_grid),
            utils.as_c_float_p(theta),
            utils.as_c_float_p(h),
            utils.as_c_float_p(v),
            utils.as_c_int(probe_grid.size),
            utils.as_c_float_p(object_grid),
            utils.as_c_int(niter))
    elif algorithm is "sirt":
        LIBTIKE.sirt.restype = utils.as_c_void_p()
        LIBTIKE.sirt(
            utils.as_c_float(object_min[0]),
            utils.as_c_float(object_min[1]),
            utils.as_c_float(object_min[2]),
            utils.as_c_float(object_size[0]),
            utils.as_c_float(object_size[1]),
            utils.as_c_float(object_size[2]),
            utils.as_c_int(ngrid[0]),
            utils.as_c_int(ngrid[1]),
            utils.as_c_int(ngrid[2]),
            utils.as_c_float_p(probe_grid),
            utils.as_c_float_p(theta),
            utils.as_c_float_p(h),
            utils.as_c_float_p(v),
            utils.as_c_int(probe_grid.size),
            utils.as_c_float_p(object_grid),
            utils.as_c_int(niter))
    else:
        raise ValueError("The {} algorithm is not an available.".format(
            algorithm))
    return object_grid


def forward(object_grid=None, object_min=None, object_size=None,
            probe_grid=None, probe_size=None,
            theta=None, h=None, v=None,
            **kwargs):
    """Forward-project probes over an object; i.e. simulate data acquisition.

    Parameters
    ----------
    probe_grid : (M, H, V, P) :py:class:`numpy.array` float
        The inital parameters of the `M` probes to be projected across
        `object_grid`. `P`, holds parameters at each grid position:

            * (..., 0) : intensity / amplitude
            * (..., 1) : relative phase shift

    Returns
    -------
    exit_probe_grid : (M, H, V, P) :py:class:`numpy.array` float
        The properties of the probe after exiting the `object_grid`.
    """
    object_grid, object_min, object_size, probe_grid, probe_size, theta, h, v \
        = _tomo_interface(object_grid, object_min, object_size,
                          probe_grid, probe_size, theta, h, v)
    # Remove zero valued probe rays
    nonzeros = (probe_grid != 0)
    th1 = theta[nonzeros]
    h1 = h[nonzeros]
    v1 = v[nonzeros]
    line_integrals = np.zeros(th1.shape, dtype=np.float32)
    # Send data to c function
    logger.info("forward {:,d} element grid".format(object_grid.size))
    logger.info("forward {:,d} rays".format(line_integrals.size))
    object_grid = utils.as_float32(object_grid)
    ngrid = object_grid.shape
    th1 = utils.as_float32(th1)
    h1 = utils.as_float32(h1)
    v1 = utils.as_float32(v1)
    line_integrals = utils.as_float32(line_integrals)
    LIBTIKE.forward_project.restype = utils.as_c_void_p()
    LIBTIKE.forward_project(
        utils.as_c_float_p(object_grid),
        utils.as_c_float(object_min[0]),
        utils.as_c_float(object_min[1]),
        utils.as_c_float(object_min[2]),
        utils.as_c_float(object_size[0]),
        utils.as_c_float(object_size[1]),
        utils.as_c_float(object_size[2]),
        utils.as_c_int(ngrid[0]),
        utils.as_c_int(ngrid[1]),
        utils.as_c_int(ngrid[2]),
        utils.as_c_float_p(th1),
        utils.as_c_float_p(h1),
        utils.as_c_float_p(v1),
        utils.as_c_int(th1.size),
        utils.as_c_float_p(line_integrals))
    exit_probe_grid = np.zeros(probe_grid.shape, dtype=np.float32)
    exit_probe_grid[nonzeros] = line_integrals
    return exit_probe_grid


def backward(object_grid, object_min, object_size,
             probe_grid, probe_size, theta, h, v,
             **kwargs):
    """Back-project a probe over an object.

    Parameters
    ----------
    probe_grid : (M, H, V, P) :py:class:`numpy.array` float
        The parameters of the `M` probes to be projected across
        `object_grid`. `P`, holds parameters at each grid position:

            * (..., 0) : 0th probe back-projection weight
            * (..., P-1) : (P-1)th probe back-projection weight

    Returns
    -------
    new_object_grid : (Z, X, Y, P) :py:class:`numpy.array` float
        An array of projection weights. The value at each grid position is the
        area of intersection of the object with the probe multiplied by the
        probe weight.
    """
    object_grid, object_min, object_size, probe_grid, probe_size, theta, h, v \
        = _tomo_interface(object_grid, object_min, object_size,
                          probe_grid, probe_size, theta, h, v)
    raise NotImplementedError()
    return new_object_grid
