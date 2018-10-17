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
This module contains the highest level functions for solving the combined
ptychotomography problem.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import logging
from tike.constants import *

__author__ = "Doga Gursoy, Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
           ]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _combined_interface(obj, obj_min,
                        data, data_min,
                        probe, theta, h, v,
                        **kwargs):
    """A function whose interface all functions in this module matches."""
    assert np.all(obj_size > 0), "Detector dimensions must be > 0."
    assert np.all(probe_size > 0), "Probe dimensions must be > 0."
    assert np.all(detector_size > 0), "Detector dimensions must be > 0."
    assert theta.size == h.size == v.size == \
        detector_grid.shape[0] == probe_grid.shape[0], \
        "The size of theta, h, v must be the same as the number of probes."
    logger.info(" _ptycho_interface says {}".format("Hello, World!"))
    return None


def admm(obj=None, obj_min=None,
         data=None, data_min=None,
         probe=None, theta=None, h=None, v=None, energy=None,
         algorithms=None,
         niter=1, rho=1, gamma=0.5,
         **kwargs):
    """Use Alternating Direction Method of Multipliers (ADMM)

    Parameters
    ----------
    obj : (Z, X, Y, P) :py:class:`numpy.array` float
        The initial guess for the reconstruction.
    obj_min : (3, ) float
        The min corner (z, x, y) of the `obj`.
    data : (M, H, V) :py:class:`numpy.array` float
        An array of detector intensities for each of the `M` probes. The
        grid of each detector is `H` pixels wide (the horizontal
        direction) and `V` pixels tall (the vertical direction).
    data_min : (2, ) float [p]
        The min corner (h, v) of the data in the global coordinate system.
    probe : (H, V) :py:class:`numpy.array` complex
        A single illumination function for the all probes.
    energy : float [keV]
        The energy of the probe
    algorithms : (2, ) string
        The names of the pytchography and tomography reconstruction algorithms.
    kwargs :
        Any keyword arguments for the pytchography and tomography
        reconstruction algorithms.
    """
    Z, X, Y = obj.shape[0:3]
    x = obj
    psi = np.ones([Z, X, Y, 2]).view(complex).flatten()
    hobj = np.ones([Z, X, Y, 2]).view(complex).flatten()
    lamda = 0j
    for i in range(niter):
        # Ptychography.
        psi = tike.ptycho.reconstruct(data=data,
                                      probe=probe, theta=theta, h=h, v=v,
                                      psi=psi,
                                      algorithm='grad',
                                      niter=1, rho=rho, gamma=gamma, reg=hobj,
                                      lamda=lamda, **kwargs)
        # Tomography.
        phi = -1j * wavelength(energy) / 2 / np.pi * np.log(psi + lamda / rho)
        x = tike.tomo.reconstruct(obj=x,
                                  theta=theta,
                                  line_integrals=phi,
                                  algorithm='grad',
                                  niter=1, **kwargs)
        # Lambda update.
        line_integrals = tike.tomo.forward(obj=x, theta=theta)
        hobj = np.exp(1j * 2 * np.pi / wavelength(energy) * line_integrals)
        lamda = lamda + rho * (psi - hobj)
        # # Update residuals.
        # r = 1 / M * np.sqrt(np.sum(np.square(np.abs(psi - hobj)),
        #                            axis=(-1, -2)))
        # s = rho * np.sqrt(np.sum(np.square(np.abs(x1 - x0)),
        #                          axis=(-1, -2)))
        # if r < epsilon_prime and s < epsilon_dual:
        #     pass
