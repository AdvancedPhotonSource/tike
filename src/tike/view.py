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
"""Define functions for plotting and viewing data of various types."""

__author__ = "Doga Gursoy, Ash Tripathi, Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

import logging
import warnings

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib import collections
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import cv2 as cv
import numpy as np
import tike.linalg

logger = logging.getLogger(__name__)


def complexHSV_to_RGB(img0):
    """Convert a complex valued array to RGB representation.

    Takes a complex valued ND array, represents the phase as hue,
    magnitude as value, and saturation as all ones in a new (..., 3) shaped
    array. This is then converted to the RGB colorspace.

    Assumes real valued inputs have a zero imaginary component.

    Parameters
    ----------
    img0 : :py:class:`numpy.array`
        A (...) shaped complex64 numpy array.

    Returns
    -------
    rgb_img : :py:class:`numpy.array`
        The (..., 3) shaped array which represents the input complex valued array
        in a RGB colorspace.
    """

    sz = img0.shape

    hsv_img = np.ones((*sz, 3), 'float32')

    hsv_img[ ..., 0 ] = np.angle( img0 )    # always scaled between +/- pi
    hsv_img[ ..., 2 ] = np.abs( img0 )      # always scaled between 0 and +inf

    #================================
    # Rescale hue to the range [0, 1]

    hsv_img[ ..., 0 ] = ( hsv_img[ ..., 0 ] + np.pi ) / ( 2 * np.pi )

    #==================================
    # convert HSV representation to RGB

    rgb_img = mplcolors.hsv_to_rgb(hsv_img)

    return rgb_img


def resize_complex_image(img0,
                         scale_factor=(1, 1),
                         interpolation=cv.INTER_LINEAR):
    """Resize a complex image via interpolation.

    Takes a M0 x N0 complex valued array, splits it up into real and imaginary,
    and resizes (interpolates) the horizontal and vertical dimensions, yielding
    a new array of size M1 x N1. The result can then be  used for further
    plotting using e.g. imshow() or imsave() from matplotlib.

    Parameters
    ----------
    img0 : :py:class:`numpy.array`
        A M0 x N0 complex64 or complex128 numpy array.
    scale_factor : 2 element positive valued float tuple,
        ( horizontal resize/scale, vertical resize/scale  )
    interpolation  : int
        cv.INTER_NEAREST  = 0, cv.INTER_LINEAR = 1
        cv.INTER_CUBIC    = 2, cv.INTER_AREA   = 3
        cv.INTER_LANCZOS4 = 4

    Returns
    -------
    imgRS : :py:class:`numpy.array`
        The new M1 x N1 which has been resized according to the scale factors
        above.
    """

    dim = (int(img0.shape[1] * scale_factor[0]),
           int(img0.shape[0] * scale_factor[1]))

    imgRS_re = cv.resize(np.real(img0), dim, interpolation)
    imgRS_im = cv.resize(np.imag(img0), dim, interpolation)

    imgRS = imgRS_re + 1j * imgRS_im

    return imgRS


def plot_probe_power(probe):
    """Draw a bar chart of relative power of each probe to the current axes.

    The power of the probe is computed as the sum of absolute squares over all
    pixels in the probe.

    Parameters
    ----------
    probe : (..., 1, 1, SHARED, WIDE, HIGH) complex64
        The probes to be analyzed.
    """
    power = np.square(tike.linalg.norm(
        probe,
        axis=(-2, -1),
        keepdims=False,
    )).flatten()
    axes = plt.gca()
    axes.bar(
        range(len(power)),
        height=power / np.sum(power),
    )
    axes.set_xlabel('Probe index')
    axes.set_ylabel('Relative probe power')


def plot_position_error(true, *args, indices=None):
    """Create a spaghetti plot of position errors.

    Parameters
    ----------
    true (N, 2) arraylike
        The true positions.
    args (N, 2) arraylike
        A sequence of positions.
    """
    errors = np.concatenate(
        [np.linalg.norm(true - p, axis=-1, keepdims=True) for p in args],
        axis=-1,
    )
    indices = np.arange(errors.shape[1]) if indices is None else indices
    plt.plot(indices, np.transpose(errors), color='k', alpha=0.1)


def _confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    if np.all(np.abs(cov) < 1e-6):
        return

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_positions_convergence(true, *args):
    """Plot position error in 2D.

    Shows the progression of scanning position movement toward the "true"
    position. Shifts the coordinates of each position, so the true positions
    are at the origin. Draws a line from a starting triangle to an ending
    circle showing the path taken between with a line.

    Parameters
    ----------
    true (N, 2)
        True scan positions; marked with a plus.
    args (N, 2)
        A sequence of positions; starts with triangle ends with a circle.

    """
    s = 5  # only show every sth point
    args = np.stack(args, axis=0)
    args = args - true
    true = np.zeros_like(true)

    keys = ['true']
    plt.scatter(
        [0],
        [0],
        marker='+',
        color='black',
    )
    if len(args) > 1:
        plt.scatter(
            args[-1][..., ::s, 0],
            args[-1][..., ::s, 1],
            marker='o',
            color='red',
            facecolor='None',
            zorder=3,
        )
        keys.append('final')

        plt.scatter(
            args[0][..., ::s, 0],
            args[0][..., ::s, 1],
            marker='^',
            color='blue',
            facecolor='None',
            zorder=2,
        )
        keys.append('initial')

    plt.axis('equal')
    plt.legend(keys)

    for i in range(len(args) - 1, 0, -1):
        lines = zip(args[i, ::s], args[i - 1, ::s])
        lc = collections.LineCollection(lines,
                                        color='black',
                                        alpha=0.1,
                                        zorder=1)
        plt.gca().add_collection(lc)

    limits = np.maximum(np.amax(np.abs(args), axis=(-3, -2)), 1)
    plt.xlim([-limits[0], limits[0]])
    plt.ylim([-limits[1], limits[1]])

    if len(args) > 1:
        _confidence_ellipse(
            args[-1][..., 0],
            args[-1][..., 1],
            plt.gca(),
            zorder=5,
            facecolor='red',
            alpha=0.1,
        )

    if len(args) > 0:
        _confidence_ellipse(
            args[0][..., 0],
            args[0][..., 1],
            plt.gca(),
            zorder=5,
            facecolor='blue',
            alpha=0.05,
        )


def plot_positions(true, *args):
    """Plot 2D positions to current axis.

    Optionally show the progression of scanning position movement. Draws a line
    from a starting triangle to an ending circle showing the path taken
    between with a line.

    Parameters
    ----------
    true (N, 2)
        True scan positions; marked with a plus.
    args (N, 2)
        A sequence of positions; starts with triangle ends with a circle.

    """
    keys = ['true']
    plt.scatter(
        true[..., 0],
        true[..., 1],
        marker='+',
        color='black',
    )
    if len(args) > 1:
        plt.scatter(
            args[-1][..., 0],
            args[-1][..., 1],
            marker='o',
            color='red',
            facecolor='None',
        )
        keys.append('current')
    if len(args) > 0:
        plt.scatter(
            args[0][..., 0],
            args[0][..., 1],
            marker='^',
            color='blue',
            facecolor='None',
        )
        keys.append('initial')
    plt.axis('equal')
    plt.legend(keys)

    if len(args) > 0:
        lines = zip(true, args[-1])
        lc = collections.LineCollection(lines, color='red', linestyle='dashed')
        plt.gca().add_collection(lc)

    for i in range(len(args) - 1, 0, -1):
        lines = zip(args[i], args[i - 1])
        lc = collections.LineCollection(lines, color='blue')
        plt.gca().add_collection(lc)


def plot_complex(Z, rmin=None, rmax=None, imin=None, imax=None):
    """Plot real and imaginary parts of a 2D image size by side.

    Takes parameters rmin, rmax, imin, imax to scale the ranges of the real
    and imaginary plots.
    """
    plt.subplot(1, 2, 1)
    plt.imshow(Z.real, vmin=rmin, vmax=rmax)
    cb0 = plt.colorbar(orientation='horizontal')
    plt.subplot(1, 2, 2)
    plt.imshow(Z.imag, vmin=imin, vmax=imax)
    cb1 = plt.colorbar(orientation='horizontal')


def plot_phase(Z, amin=None, amax=None):
    """Plot the amplitude and phase of a 2D image side by side.

    Takes parameters amin, amax to scale the range of the amplitude. The phase
    is scaled to the range -pi to pi.
    """
    amplitude, phase = np.abs(Z), np.angle(Z)
    if np.any(amplitude == 0):
        warnings.warn(
            "This phase plot will be incorrect because "
            "the phase of a zero-amplitude complex number is undefined. "
            "Adding a small constant to the amplitude may help.")
    plt.subplot(1, 2, 1)
    plt.imshow(amplitude, vmin=amin, vmax=amax)
    cb0 = plt.colorbar(orientation='horizontal')
    plt.subplot(1, 2, 2)
    plt.imshow(phase, vmin=-np.pi, vmax=np.pi, cmap=plt.cm.twilight)
    cb1 = plt.colorbar(orientation='horizontal')
    print(np.min(Z), np.max(Z))


def trajectory(x, y, connect=True, frame=None, pause=True, dt=1e-12):
    """Plot a 2D trajectory."""
    if frame is None:
        frame = [np.min(x), np.max(x), np.min(y), np.max(y)]
    ax = fig.add_subplot(111)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid('on')
    ax.axis('image')
    ax.set_xlim(frame[0], frame[1])
    ax.set_ylim(frame[2], frame[3])
    if connect is True:
        line, = ax.plot([], [], '-or')
    else:
        line, = ax.plot([], [], 'or')
    for m in range(x.size):
        line.set_xdata(np.append(line.get_xdata(), x[m]))
        line.set_ydata(np.append(line.get_ydata(), y[m]))
        plt.draw()
        if pause is True:
            plt.pause(dt)


def plot_footprint(theta, v, h):
    """Plot 2D projections of the trajectory for each pair of axes."""
    theta = theta % (np.pi) / np.pi

    ax1a = plt.subplot(1, 3, 2)
    plt.plot(h, v, color='blue', linewidth=1)
    ax1a.axis('equal')
    plt.xlabel("h")
    plt.ylabel("v")

    ax1b = plt.subplot(1, 3, 3)
    plt.scatter(theta, v, color='red', s=1)
    ax1b.axis('equal')
    plt.xticks([0, 1], [0, r'$\pi$'])
    plt.xlabel('theta')
    plt.ylabel("v")

    ax1c = plt.subplot(1, 3, 1)
    plt.scatter(h, theta, color='green', s=1)
    ax1c.axis('equal')
    plt.yticks([0, 1], [0, r'$\pi$'])
    plt.ylabel('theta')
    plt.xlabel("h")


def plot_sino_coverage(
        theta, v, h, dwell=None, bins=[16, 8, 4],
        probe_grid=[[1]], probe_shape=(0, 0)
):  # yapf: disable
    """Plot projections of minimum coverage in the sinogram space."""
    # Wrap theta into [0, pi)
    theta = theta % (np.pi)
    # Set default dwell value
    if dwell is None:
        dwell = np.ones(theta.shape)
    # Make sure probe_grid is array
    probe_grid = np.asarray(probe_grid)
    # Create one ray for each pixel in the probe grid
    dv, dh = np.meshgrid(
        np.linspace(0, probe_shape[0], probe_grid.shape[0], endpoint=False) +
        probe_shape[0] / probe_grid.shape[0] / 2,
        np.linspace(0, probe_shape[1], probe_grid.shape[1], endpoint=False) +
        probe_shape[1] / probe_grid.shape[1] / 2,
    )

    dv = dv.flatten()
    dh = dh.flatten()
    probe_grid = probe_grid.flatten()
    H = np.zeros(bins)
    for i in range(probe_grid.size):
        if probe_grid[i] > 0:
            # Compute histogram
            sample = np.stack([theta, v + dv[i]], h + dh[i], axis=1)
            dH, edges = np.histogramdd(sample,
                                       bins=bins,
                                       range=[[0, np.pi], [-.5, .5], [-.5, .5]],
                                       weights=dwell * probe_grid[i])
            H += dH
    ideal_bin_count = np.sum(dwell) * np.sum(probe_grid) / np.prod(bins)
    H /= ideal_bin_count
    # Plot
    ax1a = plt.subplot(1, 3, 2)
    plt.imshow(np.min(H, axis=0).T,
               vmin=0,
               vmax=2,
               origin="lower",
               cmap=plt.cm.RdBu)
    ax1a.axis('equal')
    plt.xticks(np.array([0, bins[1] / 2, bins[1]]) - 0.5, [-.5, 0, .5])
    plt.yticks(np.array([0, bins[2] / 2, bins[2]]) - 0.5, [-.5, 0, .5])
    plt.xlabel("h")
    plt.ylabel("v")

    ax1b = plt.subplot(1, 3, 3)
    plt.imshow(np.min(H, axis=1).T,
               vmin=0,
               vmax=2,
               origin="lower",
               cmap=plt.cm.RdBu)
    ax1b.axis('equal')
    plt.xlabel('theta')
    plt.ylabel("v")
    plt.xticks(np.array([0, bins[0]]) - 0.5, [0, r'$\pi$'])
    plt.yticks(np.array([0, bins[2] / 2, bins[2]]) - 0.5, [-.5, 0, .5])

    ax1c = plt.subplot(1, 3, 1)
    plt.imshow(np.min(H, axis=2),
               vmin=0,
               vmax=2,
               origin="lower",
               cmap=plt.cm.RdBu)
    ax1c.axis('equal')
    plt.ylabel('theta')
    plt.xlabel("h")
    plt.yticks(np.array([0, bins[0]]) - 0.5, [0, r'$\pi$'])
    plt.xticks(np.array([0, bins[1] / 2, bins[1]]) - 0.5, [-.5, 0, .5])

    return H


def plot_trajectories(theta, v, h, t):
    """Plot each trajectory as a function of time in the current figure.

    Plots two subplots in the current figure. The top one shows horizonal
    and vertical position as a function of time and the bottom shows angular
    position as a function of time.

    Returns
    -------
    ax1, ax1b : axes
        Handles to the two axes

    """
    ax1a = plt.subplot(2, 1, 1)
    plt.plot(t, h, 'c--', t, v, 'm.')
    plt.ylabel('position [cm]')
    plt.legend(['h', 'v'])
    plt.ylim([-.5, .5])
    plt.setp(ax1a.get_xticklabels(), visible=False)
    ax1b = plt.subplot(2, 1, 2, sharex=ax1a)
    plt.plot(t, theta % (2 * np.pi) / (2 * np.pi), 'yellow')
    plt.ylabel(r'theta [$2\pi$]')
    plt.xlabel('time [s]')
    plt.ylim([0, 1.])
    return ax1a, ax1b


def plot_cost_convergence(costs, times):
    """Plot a twined plot of cost vs iteration/cumulative-time

    The plot is a semi-log line plot with two lines. One line shows cost as a
    function of iteration (bottom horizontal); one line shows cost as a
    function of cumulative wall-time (top horizontal).

    Parameters
    ----------
    costs : (NUM_ITER, ) array-like
        The objective cost at each iteration.
    times : (NUM_ITER, ) array-like
        The wall-time for each iteration in seconds.

    Returns
    -------
    ax1 : matplotlib.axes._subplots.AxesSubplot
    ax2 : matplotlib.axes._subplots.AxesSubplot
    """
    ax1 = plt.subplot()

    costs = np.asarray(costs)
    alpha = 1.0 / costs.shape[1] if costs.ndim > 1 else 1.0

    color = 'black'
    ax1.semilogy()
    ax1.set_xlabel('iteration', color=color)
    ax1.set_ylabel('objective')
    ax1.plot(costs, linestyle='--', color=color, alpha=alpha)
    ax1.tick_params(axis='x', labelcolor=color)

    ax2 = ax1.twiny()

    color = 'red'
    ax2.set_xlabel('wall-time [s]', color=color)
    ax2.plot(np.cumsum(times), costs, color=color, alpha=alpha)
    ax2.tick_params(axis='x', labelcolor=color)

    return ax1, ax2


def plot_eigen_weights(weights):
    """Plot stacked line plots of probe intensity weights by position."""
    n = weights.shape[-1]

    ax1 = None
    for i in range(0, weights.shape[-1]):
        axi = plt.subplot(n, 1, i + 1, sharey=ax1)
        if i == 0:
            ax1 = axi
        axi.plot(weights[..., i])
        if i < weights.shape[-1] - 1:
            axi.set_xticklabels([])

    axi.set_xlabel('positions')
