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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from . import utils
from . import externs
import logging

__author__ = "Doga Gursoy, Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['art',
           'sirt',
           'project',
           'coverage']


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def coverage(grid_min, grid_size, ngrid, theta, h, v, line_weight=None,
             anisotropy=1):
    """Back-project lines over a grid.

    Coverage will be calculated on a grid covering the range
    `[grid_min, grid_min + grid_size)`.

    Parameters
    ----------
    grid_min : tuple float (z, x, y)
        The min corner of the grid.
    grid_size : tuple float (z, x, y)
        The side lengths of the grid along each dimension.
    ngrid : tuple int (z, x, y)
        The number of grid spaces along each dimension.
    theta, h, v : (M, ) :py:class:`numpy.array`
        The h, v, and theta coordinates of lines to back-project over
        the grid.
    line_weight : (M, ) :py:class:`numpy.array`
        Multiply the intersections lengths of the pixels and each line by these
        weights.

    Returns
    -------
    coverage_map : :py:class:`numpy.ndarray` [length * line_weight]
        An array of shape (ngrid, anisotropy) containing the sum of the
        intersection lengths multiplied by the line_weights.
    """
    grid_min = utils.as_float32(grid_min)
    grid_size = utils.as_float32(grid_size)
    ngrid = utils.as_int32(ngrid)
    assert np.all(grid_size > 0), "Grid dimensions must be > 0"
    assert np.all(ngrid > 0), "Number of grid lines must be > 0"
    h = utils.as_float32(h)
    v = utils.as_float32(v)
    theta = utils.as_float32(theta)
    if line_weight is None:
        line_weight = np.ones(theta.shape, dtype=np.float32)
    line_weight = utils.as_float32(line_weight)
    assert theta.size == h.size == v.size == line_weight.size, "line_weight," \
        " theta, h, v must be the same size"
    if anisotropy > 1:
        coverage_map = np.zeros(list(ngrid) + [anisotropy], dtype=np.float32)
    else:
        coverage_map = np.zeros(ngrid, dtype=np.float32)
        anisotropy = 1
    logging.info(" coverage {:,d} element grid".format(coverage_map.size))
    externs.c_coverage(grid_min[0], grid_min[1], grid_min[2],
                       grid_size[0], grid_size[1], grid_size[2],
                       ngrid[0], ngrid[1], ngrid[2], anisotropy,
                       theta, h, v, line_weight, h.size, coverage_map)
    return coverage_map


def project(obj, grid_min, grid_size, theta, h, v):
    """Forward-project lines over an object.

    Parameters
    ----------
    obj : (Z, X, Y) :py:class:`numpy.array`
        An array of weights to integrate each line over.
    theta, h, v : (M, ) :py:class:`numpy.array`
        The h, v, and theta coordinates of lines to integrate over `obj`.
    grid_min : tuple float (z, x, y)
        The min corner of the grid.
    grid_size : tuple float (z, x, y)
        The side lengths of the grid along each dimension.
    theta, h, v : (M, ) :py:class:`numpy.array`
        The h, v, and theta coordinates of lines to forward-project over an
        `obj.shape` grid.

    Returns
    -------
    data : (M, ) :py:class:`numpy.array`
        The integral of each line over the object.
    """
    obj = utils.as_float32(obj)
    ngrid = obj.shape
    assert np.all(np.array(grid_size) > 0), "Grid dimensions must be > 0"
    assert np.all(np.array(ngrid) > 0), "Number of grid lines must be > 0"
    theta = utils.as_float32(theta)
    h = utils.as_float32(h)
    v = utils.as_float32(v)
    assert theta.size == h.size == v.size, \
        " theta, h, v must be the same size"
    dsize = theta.size
    data = np.zeros((dsize, ), dtype=np.float32)
    externs.c_project(obj,
                      grid_min[0], grid_min[1], grid_min[2],
                      grid_size[0], grid_size[1], grid_size[2],
                      ngrid[0], ngrid[1], ngrid[2],
                      theta, h, v, dsize, data)
    return data


def art(grid_min, grid_size, data, theta, h, v, init, niter=1):
    """Reconstruct using Algebraic Reconstruction Technique (ART)

    See :cite:`gordon1970algebraic` for original description of ART.

    Parameters
    ----------
    grid_min : tuple float (z, x, y)
        The min corner of the grid.
    grid_size : tuple float (z, x, y)
        The width of the grid along each dimension.
    data : (M, ) :py:class:`np.array`
        The data for reconstruction
    theta, h, v : (M, ) :py:class:`np.array`
        The h, v, and theta coordinates of the data
    niter : int
        The number of ART iterations to perform
    init : :py:class:`np.array`
        An initial guess for reconstruction.

    Returns
    -------
    recon : :py:class:`numpy.ndarray`
        A reconstruction of grid_size.
    """
    grid_min = utils.as_float32(grid_min)
    grid_size = utils.as_float32(grid_size)
    assert np.all(grid_size > 0), "Grid dimensions must be > 0"
    data = utils.as_float32(data)
    theta = utils.as_float32(theta)
    h = utils.as_float32(h)
    v = utils.as_float32(v)
    assert theta.size == h.size == v.size == data.size, "data, theta, h, v " \
        "must be the same size"
    assert niter >= 0, "Number of iterations should be >= 0"
    init = utils.as_float32(init)
    nz, nx, ny = init.shape
    logging.info(" ART {:,d} element grid for {:,d} iterations".format(
        init.size, niter))
    externs.c_art(grid_min[0], grid_min[1], grid_min[2],
                  grid_size[0], grid_size[1], grid_size[2],
                  nz, nx, ny,
                  data, theta, h, v, data.size, init, niter)
    return init


def sirt(grid_min, grid_size, data, theta, h, v, init, niter=1):
    """Reconstruct using Simultaneous Iterative Reconstruction Technique (SIRT)

    Parameters
    ----------
    grid_min : tuple float (z, x, y)
        The min corner of the grid.
    grid_size : tuple float (z, x, y)
        The width of the grid along each dimension.
    data : (M, ) :py:class:`np.array`
        The data for reconstruction
    theta, h, v : (M, ) :py:class:`np.array`
        The h, v, and theta coordinates of the data
    niter : int
        The number of SIRT iterations to perform
    init : :py:class:`np.array`
        An initial guess for reconstruction.

    Returns
    -------
    recon : :py:class:`numpy.ndarray`
        A reconstruction of grid_size.
    """
    grid_min = utils.as_float32(grid_min)
    grid_size = utils.as_float32(grid_size)
    assert np.all(grid_size > 0), "Grid dimensions must be > 0"
    data = utils.as_float32(data)
    theta = utils.as_float32(theta)
    h = utils.as_float32(h)
    v = utils.as_float32(v)
    assert theta.size == h.size == v.size == data.size, "data, theta, h, v " \
        "must be the same size"
    assert niter >= 0, "Number of iterations should be >= 0"
    init = utils.as_float32(init)
    nz, nx, ny = init.shape
    logging.info(" SIRT {:,d} element grid for {:,d} iterations".format(
        init.size, niter))
    externs.c_sirt(grid_min[0], grid_min[1], grid_min[2],
                   grid_size[0], grid_size[1], grid_size[2],
                   nz, nx, ny,
                   data, theta, h, v, data.size, init, niter)
    return init
