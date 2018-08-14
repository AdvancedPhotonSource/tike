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
This module contains functions for solving the ptychography problem.

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
Each function in this module should have the following interface:

Parameters
----------
detector_grid : (M, H, V) :py:class:`numpy.array` float
    An array of detector intensities for each of the `M` probes. The
    grid of each detector is `H` pixels wide (the horizontal
    direction) and `V` pixels tall (the vertical direction).
detector_min : (2, ) float
    The min corner (h, v) of `detector_grid`.
detector_size : (2, ) float
    The side lengths (h, v) of `detector_grid` along each dimension.
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
    Keyword arguments specific to this function.

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
from . import externs
import logging

__author__ = "Doga Gursoy, Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ["reconstruct",
           "propagate_forward",
           "propagate_backward",
           ]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ptycho_interface(detector_grid, detector_min, detector_size,
                      probe_grid, probe_size, theta, h, v,
                      **kwargs):
    """A function whose interface all functions in this module matchesself.

    This function also sets default values for functions in this module.
    """
    if detector_grid is None:
        raise ValueError()
    if detector_min is None:
        object_min = (-0.5, -0.5)
    if detector_size is None:
        object_size = (1.0, 1.0)
    if probe_grid is None:
        raise ValueError()
    if probe_size is None:
        probe_size = (1.0, 1.0)
    if theta is None:
        raise ValueError()
    if h is None:
        h = np.full(theta.shape, -0.5)
    if v is None:
        v = np.full(theta.shape, -0.5)
    assert np.all(detector_size > 0), "Detector dimensions must be > 0."
    assert np.all(probe_size > 0), "Probe dimensions must be > 0."
    assert theta.size == h.size == v.size == \
        detector_grid.shape[0] == probe_grid.shape[0], \
        "The size of theta, h, v must be the same as the number of probes."
    # logging.info(" _ptycho_interface says {}".format("Hello, World!"))
    return (detector_grid, detector_min, detector_size,
            probe_grid, probe_size, theta, h, v)


def reconstruct(detector_grid=None, detector_min=None,
                detector_size=None,
                probe_grid=None, probe_size=None,
                theta=None, h=None, v=None,
                algorithm=None, **kwargs):
    """Reconstruct the `probe_grid` using the given `algorithm`.

    Parameters
    ----------
    probe_grid : (M, H, V, P) :py:class:`numpy.array` float
        The initial guess for the reconstruction.

    Returns
    -------
    new_probe_grid : (M, H, V, P) :py:class:`numpy.array` float
        The updated parameters of the `M` probes.
    """
    detector_grid, detector_min, detector_size, probe_grid, probe_size, theta,\
        h, v = _ptycho_interface(detector_grid, detector_min, detector_size,
                                 probe_grid, probe_size, theta, h, v)
    raise NotImplementedError()
    return new_probe_grid


def propagate_forward(detector_grid=None, detector_min=None,
                      detector_size=None,
                      probe_grid=None, probe_size=None,
                      theta=None, h=None, v=None,
                      **kwargs):
    """A function whose interface all functions in this module matches."""
    detector_grid, detector_min, detector_size, probe_grid, probe_size, theta,\
        h, v = _ptycho_interface(detector_grid, detector_min, detector_size,
                                 probe_grid, probe_size, theta, h, v)
    logging.info(" forward-propagate {} {} by {} probes.".format(theta.size,
                 probe_grid.shape[0], probe_grid.shape[1]))
    raise NotImplementedError()
    return new_detector_grid


def propagate_backward(detector_grid=None, detector_min=None,
                       detector_size=None,
                       probe_grid=None, probe_size=None,
                       theta=None, h=None, v=None,
                       **kwargs):
    """A function whose interface all functions in this module matches."""
    detector_grid, detector_min, detector_size, probe_grid, probe_size, theta,\
        h, v = _ptycho_interface(detector_grid, detector_min, detector_size,
                                 probe_grid, probe_size, theta, h, v)
    logging.info(" back-propagate {} {} by {} probes.".format(theta.size,
                 probe_grid.shape[0], probe_grid.shape[1]))
    raise NotImplementedError()
    return new_probe_grid
