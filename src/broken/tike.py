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
"""Define the highest level functions for solving ptycho-tomography problem."""

__author__ = "Doga Gursoy, Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    'admm',
    'simulate',
]

import logging

import numpy as np

from tike.communicator import MPICommunicator
from tike.constants import *
import tike.ptycho
import tike.tomo

logger = logging.getLogger(__name__)


def _combined_interface(
        obj,
        data,
        probe, theta, v, h,
        **kwargs
):  # yapf:disable
    """Define an interface that all functions in this module match."""
    assert np.all(obj_size > 0), "Detector dimensions must be > 0."
    assert np.all(probe_shape > 0), "Probe dimensions must be > 0."
    assert np.all(detector_shape > 0), "Detector dimensions must be > 0."
    assert theta.size == h.size == v.size == \
        detector_grid.shape[0] == probe_grid.shape[0], \
        "The size of theta, h, v must be the same as the number of probes."
    logger.info(" _ptycho_interface says {}".format("Hello, World!"))
    return None


def admm(
        obj=None, voxelsize=1.0,
        data=None,
        probe=None, theta=None, scan=None, energy=None,
        num_iter=1, rho=0.5, pkwargs=None, tkwargs=None,
        comm=None,
):  # yapf:disable
    """Solve using the Alternating Direction Method of Multipliers (ADMM).

    Parameters
    ----------
    obj : (Z, X, Y, P) :py:class:`numpy.array` float
        The initial guess for the reconstruction.
    voxelsize : float [cm]
        The side length of an `obj` voxel.
    data : (M, H, V) :py:class:`numpy.array` float
        An array of detector intensities for each of the `M` probes. The
        grid of each detector is `H` pixels wide (the horizontal
        direction) and `V` pixels tall (the vertical direction).
    probe : (H, V) :py:class:`numpy.array` complex
        A single illumination function for the all probes.
    energy : float [keV]
        The energy of the probe
    algorithms : (2, ) string
        The names of the pytchography and tomography reconstruction algorithms.
    num_iter : int
        The number of ADMM interations.
    pkwargs : dict
        Arguments to pass to the tike.ptycho.reconstruct.
    tkwargs : dict
        Arguments to pass to the tike.tomo.reconstruct.

    """
    comm = MPICommunicator() if comm is None else comm
    # Supress logging in the ptycho and tomo modules
    plog = logging.getLogger('tike.ptycho')
    tlog = logging.getLogger('tike.tomo')
    log_levels = [plog.level, tlog.level]
    plog.setLevel(logging.WARNING)
    tlog.setLevel(logging.WARNING)
    # Ptychography setup
    pkwargs = {
    'algorithm': 'cgrad',
    'num_iter': 1,
    } if pkwargs is None else pkwargs
    psi = np.ones(
        [
            len(data),  # The number of views.
            np.sum(comm.allgather(obj.shape[0])),  # The height of psi.
            obj.shape[2],  # The width of psi.
        ],
        dtype=np.complex64)
    logger.debug("psi shape is {}".format(psi.shape))
    hobj = np.ones_like(psi)
    lamda = np.zeros_like(psi)
    # Tomography setup
    tkwargs = {
        'algorithm': 'cgrad',
        'num_iter': 1,
    } if tkwargs is None else tkwargs
    # Start ADMM
    for i in range(num_iter):
        logger.info("ADMM iteration {}".format(i))
        # Ptychography
        reg = hobj + lamda / rho
        result = tike.ptycho.reconstruct(
            data=data,
            probe=probe,
            scan=scan,
            psi=psi,
            rho=rho, reg=reg,
            **pkwargs
        )  # yapf: disable
        psi = result['psi']
        # Tomography.
        phi = -1j / wavenumber(energy) * np.log(psi + lamda / rho) / voxelsize
        # Tomography
        phi = comm.get_tomo_slice(phi)
        result = tike.tomo.reconstruct(
            obj=obj, theta=theta, integrals=phi, **tkwargs)
        obj = result['obj']
        # Lambda update.
        line_integrals = tike.tomo.simulate(obj=obj, theta=theta) * voxelsize
        hobj = np.exp(1j * wavenumber(energy) * line_integrals)
        hobj = comm.get_ptycho_slice(hobj)
        # lamda = lamda + rho * (hobj - psi)
    # Restore logging in the tomo and ptycho modules
    logging.getLogger('tike.ptycho').setLevel(log_levels[0])
    logging.getLogger('tike.tomo').setLevel(log_levels[1])
    return obj


def simulate(
        obj, voxelsize,
        probe, theta, scan, energy,
        detector_shape,
        comm=None
):  # yapf:disable
    """Simulate data acquisition from an object, probe, and positions."""
    comm = MPICommunicator() if comm is None else comm
    # Tomography simulation
    line_integrals = tike.tomo.simulate(obj=obj, theta=theta) * voxelsize
    psi = np.exp(1j * wavenumber(energy) * line_integrals)
    psi = comm.get_ptycho_slice(psi)
    # Ptychography simulation
    data = tike.ptycho.simulate(
        detector_shape=int(detector_shape),
        probe=probe,
        scan=scan,
        psi=psi,
    )
    return data
