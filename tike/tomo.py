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

import numpy as np
from . import utils
from . import externs
import logging


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['art',
           'project',
           'coverage']


logger = logging.getLogger(__name__)


def coverage(obj, x, y, theta):
    obj = utils.as_float32(obj)
    x = utils.as_float32(x)
    y = utils.as_float32(y)
    theta = utils.as_float32(theta)
    ox, oy, oz = obj.shape
    dsize = theta.size
    cov = np.zeros(obj.shape, dtype=np.float32)
    externs.c_coverage(ox, oy, oz, x, y, theta, dsize, cov)
    return cov


def coverage_approx(procedure, region, pixel_size, line_width,
                    anisotropy=False):
    """Approximate procedure coverage with thick lines.

    The intersection between each line and each pixel is approximated by
    the product of `line_width**2` and the length of segment of the
    line segment `alpha` which passes through the pixel along the line.

    If `anisotropy` is `True`, then `coverage_map.shape` is `(L, M, N, 2, 2)`,
    where the two extra dimensions contain coverage anisotopy information as a
    second order tensor.

    Parameters
    ----------
    procedure : list of :py:class:`np.array`
        Each element of 'procedure' is a (7,) array which describes a series
        of lines as `[x1, y1, z1, x2, y2, z2, weight]`. Presently, `z1`
        must equal `z2`.
    line_width` : float
        The side length of the square cross-section of the line.
    region : :py:class:`np.array`
        A box in which to map the coverage. Specify the bounds as
        `[[min_x, max_x], [min_y, max_y], [min_z, max_z]]`.
        i.e. column vectors pointing to the min and max corner.
    pixel_size : float
        The edge length of the pixels in the coverage map in centimeters.
    anisotropy : bool
        Determines whether the coverage map includes anisotropy information.

    Returns
    -------
    coverage_map : :py:class:`numpy.ndarray`
        A discretized map of the approximated procedure coverage.

    Raises
    ------
    ValueError when lines have have non-constant z coordinate.
    """
    box = np.asanyarray(region)
    # Define the locations of the grid lines (gx, gy, gz)
    gx = np.arange(box[0, 0], box[0, 1] + pixel_size, pixel_size)
    gy = np.arange(box[1, 0], box[1, 1] + pixel_size, pixel_size)
    gz = np.arange(box[2, 0], box[2, 1] + pixel_size, pixel_size)
    # the number of pixels = number of gridlines - 1
    sx, sy, sz = gx.size-1, gy.size-1, gz.size-1
    # Preallocate the coverage_map
    if anisotropy:
        coverage_map = np.zeros((sx, sy, sz, 2, 2))
    else:
        coverage_map = np.zeros((sx, sy, sz))
    for line in procedure:
        # line -> x1, y1, z1, x2, y2, z2, weight
        x0, y0, z0 = line[0], line[1], line[2]
        x1, y1, z1 = line[3], line[4], line[5]
        weight = line[6]
        # avoid upper-right boundary errors
        if (x1 - x0) == 0:
            x0 += 1e-12
        if (y1 - y0) == 0:
            y0 += 1e-12
        if (z1 - z0) != 0:
            raise ValueError("Lines must have constant z coordinate.")
        # vector lengths (ax, ay)
        ax = (gx - x0) / (x1 - x0)
        ay = (gy - y0) / (y1 - y0)
        # layer weights in the z direction
        az = np.maximum(0, (np.minimum(z1 + line_width/2, (gz + pixel_size))
                            - np.maximum(z0 - line_width/2, gz)) / line_width)
        # edges of alpha (a0, a1)
        ax0 = min(ax[0], ax[-1])
        ax1 = max(ax[0], ax[-1])
        ay0 = min(ay[0], ay[-1])
        ay1 = max(ay[0], ay[-1])
        a0 = max(max(ax0, ay0), 0)
        a1 = min(min(ax1, ay1), 1)
        # sorted alpha vector
        cx = (ax >= a0) & (ax <= a1)
        cy = (ay >= a0) & (ay <= a1)
        alpha = np.sort(np.r_[ax[cx], ay[cy]])
        if len(alpha) > 0:
            # lengths
            xv = x0 + alpha * (x1 - x0)
            yv = y0 + alpha * (y1 - y0)
            lx = np.ediff1d(xv)
            ly = np.ediff1d(yv)
            dist = np.sqrt(lx**2 + ly**2)
            dist2 = np.dot(dist, dist)
            ind = dist != 0
            # indexing
            mid = alpha[:-1] + np.ediff1d(alpha) / 2.
            xm = x0 + mid * (x1 - x0)
            ym = y0 + mid * (y1 - y0)
            ix = np.floor(np.true_divide(sx * (xm - box[0, 0]),
                                         sx * pixel_size)).astype('int')
            iy = np.floor(np.true_divide(sy * (ym - box[0, 1]),
                                         sy * pixel_size)).astype('int')
            iz = np.arange(sz + 1)[az > 0].astype('int')
            try:
                magnitude = dist * line_width**2 * weight
                if anisotropy:
                    beam = line[0:2] - line[3:5]
                    beam_angle = np.arctan2(beam[1], beam[0])
                    tensor = tensor_at_angle(beam_angle, magnitude)
                    for i in iz:
                        coverage_map[ix, iy, i, :, :] += tensor * az[i]
                else:
                    for i in iz:
                        coverage_map[ix, iy, i] += magnitude * az[i]
            except IndexError as e:
                warnings.warn("{}\nix is {}\niy is {}\niz is {}".format(e, ix,
                              iy, iz), RuntimeWarning)
    return coverage_map / pixel_size**3


def project(obj, x, y, theta):
    obj = utils.as_float32(obj)
    x = utils.as_float32(x)
    y = utils.as_float32(y)
    theta = utils.as_float32(theta)
    ox, oy, oz = obj.shape
    dsize = theta.size
    data = np.zeros((dsize, ), dtype=np.float32)
    externs.c_project(obj, ox, oy, oz, x, y, theta, dsize, data)
    return data


def art(data, x, y, theta):
    data = utils.as_float32(data)
    x = utils.as_float32(x)
    y = utils.as_float32(y)
    theta = utils.as_float32(theta)
    recon = np.ones((100, 100, 100), dtype=np.float32)
    externs.c_art(data, x, y, theta, recon)
    return recon
