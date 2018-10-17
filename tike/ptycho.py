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
This module contains functions for solving the phase-retrieval problem.

Coordinate Systems
==================
`h, v` are the horizontal and vertical directions perpendicular
to the probe direction where positive directions are to the right and up
respectively.

Functions
=========
Each function in this module should have the following interface:

Parameters
----------
data : (M, H, V) :py:class:`numpy.array` float
    An array of detector intensities for each of the `M` probes. The
    grid of each detector is `H` pixels wide (the horizontal
    direction) and `V` pixels tall (the vertical direction).
probe : (H, V) :py:class:`numpy.array` complex
    The single illumination function of the `M` probes.
psi : (T, H, V) :py:class:`numpy.array` complex
    The object transmission function for each of the `T` views.
foo_min : (2, ) float [p]
    The min corner (h, v) of `foo` in the global coordinate system. `foo`
    could be `data`, `psi`, etc.
kwargs
    Keyword arguments specific to this function. `**kwargs` should always be
    included so that extra parameters are ignored instead of raising an error.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import logging

__author__ = "Doga Gursoy, Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ["reconstruct",
           "simulate",
           ]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ptycho_interface(data, data_min,
                      probe, theta, h, v,
                      psi, psi_min, **kwargs):
    """A function whose interface all functions in this module matchesself.

    This function also sets default values for functions in this module.
    """
    if data is None:
        raise ValueError()
    if data_min is None:
        data_min = (-0.5, -0.5)
    if probe is None:
        raise ValueError()
    if theta is None:
        raise ValueError()
    if h is None:
        raise ValueError()
    if v is None:
        raise ValueError()
    if psi is None:
        raise ValueError()
    if psi_min is None:
        psi_min = (-0.5, -0.5)
    assert theta.size == h.size == v.size == data.shape[0] == \
        "The size of theta, h, v must be the same as the number of data."
    # logger.info(" _ptycho_interface says {}".format("Hello, World!"))
    return (data, data_min,
            probe, theta, h, v)


def locate_pad(pshape, ushape):
    """Return the min and max indices of the range u in range p."""
    xmin = (pshape - ushape) // 2
    xmax = xmin + ushape
    return xmin, xmax


def gaussian(size, rin=0.8, rout=1.0):
    """Return a complex gaussian probe distribution

    Illumination probe represented on a 2D regular grid.

    A finite-extent circular shaped probe is represented as
    a complex wave. The intensity of the probe is maximum at
    the center and damps to zero at the borders of the frame.

    Parameters
    ----------
    size : int
        The side length of the distribution
    rin : float [0, 1) < rout
        The inner radius of the distribution where the dampening of the
        intensity will start.
    rout : float (0, 1] > rin
        The outer radius of the distribution where the intensity will reach
        zero.
    """
    r, c = np.mgrid[:size, :size] + 0.5
    rs = np.sqrt((r - size/2)**2 + (c - size/2)**2)
    rmax = np.sqrt(2) * 0.5 * rout * rs.max() + 1.0
    rmin = np.sqrt(2) * 0.5 * rin * rs.max()
    img = np.zeros((size, size), dtype='float32')
    img[rs < rmin] = 1.0
    img[rs > rmax] = 0.0
    zone = np.logical_and(rs > rmin, rs < rmax)
    img[zone] = np.divide(rmax - rs[zone], rmax - rmin)
    return img


def pad_grid(padded_shape=None, padded_min=None,
             unpadded_grid=None, h=None, v=None):
    """Pad the `unpadded_grid` with zeros to match the `padded_shape`.

    The `unpadded_grid` keeps its own data type. The padded_shape is the same
    in all dimensions except for the last two. The unpadded_grid is extracted
    or placed into the an array that is the size of the padded_shape.

    Notes
    -----
    Issue (numpy/numpy #11126, 21 May 2018): Copying into preallocated array
    is faster because of the way that np.pad uses concatenate.
    """
    assert np.all(list(padded_shape[-2:]) >= list(unpadded_grid.shape[-2:])), \
        "Padded shape is smaller than unpadded shape!"
    padded_grid = np.zeros(padded_shape, dtype=unpadded_grid.dtype)
    hmin, hmax = locate_pad(padded_shape[-2], unpadded_grid.shape[-2])
    vmin, vmax = locate_pad(padded_shape[-1], unpadded_grid.shape[-1])
    padded_grid[..., hmin:hmax, vmin:vmax] = unpadded_grid
    return padded_grid


def unpad_grid(padded_grid=None, padded_min=None,
               unpadded_shape=None, h=None, v=None):
    """Crop the `padded_grid` to match the `unpadded_shape`.

    The `padded_grid` keeps its own data type. The padded_grid is the same
    in all dimensions except for the last two. The unpadded_grid is extracted
    or placed into the an array that is the size of the padded_grid.
    """
    assert np.all(list(padded_grid.shape[-2:]) >= list(unpadded_shape[-2:])), \
        "Padded shape is smaller than unpadded shape!"
    hmin, hmax = locate_pad(padded_grid.shape[-2], unpadded_shape[-2])
    vmin, vmax = locate_pad(padded_grid.shape[-1], unpadded_shape[-1])
    return padded_grid[..., hmin:hmax, vmin:vmax]


def float_shift(a, shift):
    """Use linear interpolation to shift an array less than 1 index.

    The shift is truncated to whatever is after the decimal. For example
    -2.3 becomes -0.3 and 23 becomes 0.

    Parameters
    ----------
    a : array_like
        Input array.
    shift : array_like
        The distance to shift in each dimension.

    Return
    ------
    lo : ndarray
        Output array, one unit larger than `a`

    Example
    -------
    A = np.ones([3])
    B = float_shift(A, [-0.2])
    B
    array([ 0.2,  1. ,  1. ,  0.8])
    """
    shift = np.asanyarray(shift, dtype=float) % 1.0
    lo = np.pad(a, [0, 1], mode='constant')
    for i in range(shift.size):
        hi = np.roll(lo, 1, axis=i)
        lo = lo + shift[i]*(hi-lo)
    return lo


def combine_grids(grids, T, h, v,
                  combined_shape, combined_min):
    """Combines some grids by summation.

    The grids are interpolated onto the combined grid using bilinear
    interpolation.

    Parameters
    ----------
    grids : (M, H, V) :py:class:`numpy.array`
        The values on the grids
    h, v : (M, ) :py:class:`numpy.array` float [m]
        The coordinates of the minimum corner of the M grids
    T : (M, ) :py:class:`numpy.array` int
        An index to separate the grids
    combined_shape : (N+2, ) int
        The last two are numbers of the tuple are the number of indices along
        the h and v directions of the combined grid.
    combined_min : (2, ) float [m]
        The coordinates of the minimum corner of the combined grids

    Return
    ------
    combined : (T, H, V) :py:class:`numpy.array`
        The combined grid
    """
    # TODO: assert grid resolution is the same for grids and combined
    m_shape, h_shape, v_shape = grids.shape
    ch_shape, cv_shape = combined_shape[-2:]

    # Find integer indices of the min corner of each grid
    H = h // 1
    V = v // 1
    Hmin = combined_min[-2] // 1
    Vmin = combined_min[-1] // 1
    # Shift the integer indices such that the min corner of all grids is zero
    T = (T).astype(int)
    H = (H - Hmin).astype(int)
    H1 = (H + h_shape).astype(int)
    V = (V - Vmin).astype(int)
    V1 = (V + v_shape).astype(int)
    # Create a summed_grids large enough to hold all of the grids
    # Assume that the overlapping grids are not sparse
    ct_shape = np.max(T) + 1
    combined = np.zeros([ct_shape, ch_shape, cv_shape], dtype=grids.dtype)
    # Add each of the grids to the appropriate place on the summed_grids
    for M in range(m_shape):
        combined[T[M], H[M]:H1[M], V[M]:V1[M]] += grids[M, :, :]
    return combined


def uncombine_grids(grid_shape, T, h, v,
                    combined, combined_min):
    """Extract a series of grids from a single grid.

    The grids are interpolated onto the combined grid using bilinear
    interpolation.

    Parameters
    ----------
    grid_shape : (N+2, ) int
        The last two are numbers of the tuple are the number of indices along
        the h and v directions of the grids.
    T : (M, ) :py:class:`numpy.array` int
        An index to separate the grids.
    h, v : (M, ) :py:class:`numpy.array` float [m]
        The real coordinates of the minimum corner of the M grids.

    combined : (T, H, V) :py:class:`numpy.array`
        The combined grids.
    combined_min : (2, ) float [m]
        The real coordinates of the minimum corner of the combined grids

    Return
    ------
    grids : (M, H, V) :py:class:`numpy.array`
        The decombined grids
    """
    # TODO: assert grid resolution is the same for grids and combined
    m_shape = T.size
    h_shape, v_shape = grid_shape[-2:]
    ch_shape, cv_shape = combined.shape[-2:]
    # Find integer indices of the min corner of each grid
    H = h // 1
    V = v // 1
    Hmin = combined_min[-2] // 1
    Vmin = combined_min[-1] // 1
    # Shift the integer indices such that the min corner of all grids is zero
    T = (T).astype(int)
    H = (H - Hmin).astype(int)
    H1 = (H + h_shape).astype(int)
    V = (V - Vmin).astype(int)
    V1 = (V + v_shape).astype(int)
    # Create a summed_grids large enough to hold all of the grids
    # Assume that the overlapping grids are not sparse
    grids = np.zeros([m_shape, h_shape, v_shape], dtype=combined.dtype)
    # Retrive the updated values of each of the grids
    for M in range(m_shape):
        grids[M, :, :] = combined[T[M], H[M]:H1[M], V[M]:V1[M]]
    return grids


def grad(data=None, data_min=None,
         probe=None, theta=None, h=None, v=None,
         psi=None, psi_min=None,
         reg=(1+0j), niter=1, rho=0.5, gamma=0.25, lamda=0j, epsilon=1e-8,
         **kwargs):
    """Use gradient descent to update estimates for `psi`, the object
    transmission function.

    Parameters
    ----------
    reg : (T, H, V, P) :py:class:`numpy.array` complex
        The regularizer for psi.
    rho : float
        The positive penalty parameter. TODO@Selin Create better description
    gamma : float
        The ptychography gradient descent step size.
    lamda : float
        The dual variable. TODO:@Selin Create better description
    epsilon : float
        Primal residual absolute termination criterion.
        TODO:@Selin Create better description
    """
    if not (np.iscomplexobj(psi) and np.iscomplexobj(probe)
            and np.iscomplexobj(reg)):
        raise TypeError("psi, probe, and reg must be complex.")
    # Compute padding between probe and detector size
    npadx = (data.shape[1] - probe.shape[0]) // 2
    npady = (data.shape[2] - probe.shape[1]) // 2
    # Compute probe inverse
    # TODO: Update the probe too
    probe_inverse = np.conj(probe)
    wavefronts = np.empty([h.size, probe.shape[0], probe.shape[1]],
                          dtype='complex')
    for i in range(niter):
        upd_psi = np.zeros(psi.shape, dtype='complex')
        # combine all wavefronts into one array
        for m in range(h.size):
            wavefronts[m] = psi[theta[m],
                                h[m]:h[m] + probe.shape[0],
                                v[m]:v[m] + probe.shape[1]]
        # Compute near-plane wavefront
        nearplane = probe * wavefronts
        # Pad before FFT
        nearplane_pad = np.pad(nearplane,
                               ((0, 0), (npadx, npadx), (npady, npady)),
                               mode='constant')
        # Go far-plane
        farplane = np.fft.fft2(nearplane_pad)
        # Replace the amplitude with the measured amplitude.
        farplane = np.sqrt(data) * np.exp(1j * np.angle(farplane))
        # Back to near-plane.
        new_nearplane = np.fft.ifft2(farplane)[...,
                                               npadx:npadx+probe.shape[0],
                                               npady:npady+probe.shape[1]]
        # Update measurement patch.
        upd_m = probe_inverse * (new_nearplane - nearplane)
        # Combine measurement with other updates
        for m in range(h.size):
            upd_psi[theta[m],
                    h[m]:h[m]+probe.shape[0],
                    v[m]:v[m]+probe.shape[1]] += upd_m[m]
        # Update psi
        psi = ((1 - gamma * rho) * psi
               + gamma * rho * (reg - lamda / rho)
               + (gamma / 2) * upd_psi)

    return psi


def pad(phi, padded_shape):
    """Pads phi according to detector size."""
    npadx = (padded_shape[0] - phi.shape[1]) // 2
    npady = (padded_shape[1] - phi.shape[2]) // 2
    return np.pad(phi, ((0, 0), (npadx, npadx), (npady, npady)),
                  mode='constant')


def exitwave(prb, psi, theta, h, v):
    """Compute the wavefront from probe function and track of psi"""
    wave = np.zeros([h.size] + list(prb.shape), dtype=complex)
    for m in range(h.size):
        wave[m] = prb * psi[theta[m],
                            h[m]:h[m] + prb.shape[0],
                            v[m]:v[m] + prb.shape[1]]
    return wave


def simulate(data_shape=None, data_min=None,
             probe=None, theta=None, h=None, v=None,
             psi=None, psi_min=None,
             **kwargs):
    """Propagate the wavefront to the detector."""
    phi = pad(exitwave(probe, psi, theta, h, v), data_shape)
    intensity = np.square(np.abs(np.fft.fft2(phi)))
    return intensity.astype('float')


def reconstruct(data=None, data_min=None,
                probe=None, theta=None, h=None, v=None,
                psi=None, psi_min=None,
                algorithm=None, niter=1, **kwargs):
    """Reconstruct the `psi` and `probe` using the given `algorithm`.

    Parameters
    ----------
    probe : (H, V, P) :py:class:`numpy.array` float
        The initial guess for the illumnination function of each measurement.
    psi : (T, H, V, P) :py:class:`numpy.array` float
        The inital guess of the object transmission function at each angle.
    algorithm : string
        The name of one of the following algorithms to use for reconstructing:

            * grad : gradient descent

    Returns
    -------
    new_probe : (M, H, V, P) :py:class:`numpy.array` float
        The updated illumination function of each measurement.
    new_psi : (T, H, V, P) :py:class:`numpy.array` float
        The updated obect transmission function at each angle.
    """
    data, data_min, probe, theta, h, v, psi, psi_min = \
        _ptycho_interface(data, data_min,
                          probe, theta, h, v,
                          psi, psi_min, **kwargs)
    # Send data to c function
    logger.info("{} on {:,d} element grid for {:,d} iterations".format(
                algorithm, data.size, niter))
    # Add new algorithms here
    # TODO: The size of this function may be reduced further if all recon clibs
    #   have a standard interface. Perhaps pass unique params to a generic
    #   struct or array.
    if algorithm is "grad":
        new_psi = grad(data=data,
                       probe=probe.view(complex).squeeze(),
                       niter=niter, **kwargs)
        assert np.iscomplexobj(new_psi)
        new_psi = new_psi.view(float).reshape(
            line_integrals.shape)
    else:
        raise ValueError("The {} algorithm is not an available.".format(
            algorithm))
    return new_psi
