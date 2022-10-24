#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test functions in tike.trajectory."""

from signal import Handlers
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.colors as mplcolors
import cv2 as cv
import tike.view
import os.path

__author__ = "Ash Tripathi, "
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

###################################################################################################


def create_color_wheel(nxny, Hmin=-np.pi, Hmax=+np.pi):

    #=================================================================
    # numerically create the value and hue components of a color wheel
    #=================================================================

    nx, ny = nxny

    xx, yy = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))

    V = np.sqrt(xx**2 + yy**2)
    V[V > 1] = 0

    H = np.arctan2(xx, yy)

    H = np.angle(
        np.exp(1j * (H - np.pi / 2))
    )  # global phase offset so that 0 and +\- pi are on the horizontal axis

    #============================================
    # zero out colors between certain phase range
    #============================================

    H[(H > Hmax) | (H < Hmin)] = 0

    # plt.figure()
    # plt.imshow( H )

    #================================================================
    # assign the value and hue componenets to a ny x nx x 3 HSV image
    #================================================================

    hsv_img = np.ones((ny, nx, 3), 'float32')

    # Rescale hue to the range [0, 1]
    hsv_img[..., 0] = (H + np.pi) / (np.pi * 2)
    hsv_img[..., 2] = V

    #==================================
    # convert HSV representation to RGB
    #==================================

    rgb_img = mplcolors.hsv_to_rgb(hsv_img)

    # plt.figure()
    # plt.imshow(rgb_img)

    return rgb_img, H, V


###################################################################################################


def create_color_rectangle(sxsy, Hmin=-np.pi, Hmax=+np.pi):

    #=====================================================================
    # numerically create the value and hue components of a color rectangle
    #=====================================================================

    szx, szy = sxsy

    V = np.tile(np.linspace(0, 1, szx), (szy, 1))

    H = np.linspace(-np.pi, np.pi, szy).reshape(szy, 1)
    H = np.tile(H, (1, szx))

    #============================================
    # zero out colors between certain phase range
    #============================================

    H[(H > Hmax) | (H < Hmin)] = 0

    # plt.figure()
    # plt.imshow( H )

    #================================================================
    # assign the value and hue componenets to a ny x nx x 3 HSV image
    #================================================================

    hsv_img = np.ones((szy, szx, 3), 'float32')

    # Rescale hue to the range [0, 1]
    hsv_img[..., 0] = (H + np.pi) / (np.pi * 2)
    hsv_img[..., 2] = V

    #==================================
    # convert HSV representation to RGB
    #==================================

    color_rectangle = mplcolors.hsv_to_rgb(hsv_img)

    # plt.figure()
    # plt.imshow(color_rectangle)

    return color_rectangle, H, V


###################################################################################################


def test_resize_complex_image():

    nxny = (31, 40)

    resize_factor_xy = (2.432, 3.867)

    cv_interp = [
        cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_AREA,
        cv.INTER_LANCZOS4
    ]

    #=====================================
    # create a random complex valued array
    #=====================================

    phi = (2 * np.random.rand(nxny[1], nxny[0]).astype(np.float32) - 1
          ) + 1j * (2 * np.random.rand(nxny[1], nxny[0]).astype(np.float32) - 1)

    #=================================================================
    # resize phi using different built in openCV interpolation methods
    #=================================================================

    final_size_xy = (int(nxny[0] * resize_factor_xy[0]),
                     int(nxny[1] * resize_factor_xy[1]))

    for x in cv_interp:
        imgRSx = tike.view.resize_complex_image(phi, resize_factor_xy, x)
        assert (imgRSx.shape[0] == final_size_xy[1])
        assert (imgRSx.shape[1] == final_size_xy[0])
        assert (imgRSx.dtype == phi.dtype)

    #===============================================
    # resize phi using function default combinations
    #===============================================

    imgRSx = tike.view.resize_complex_image(phi)
    assert (imgRSx.shape[0] == nxny[1])
    assert (imgRSx.shape[1] == nxny[0])
    assert (imgRSx.dtype == phi.dtype)

    imgRSx = tike.view.resize_complex_image(phi, interpolation=cv.INTER_CUBIC)
    assert (imgRSx.shape[0] == nxny[1])
    assert (imgRSx.shape[1] == nxny[0])
    assert (imgRSx.dtype == phi.dtype)

    final_size_xy = (int(nxny[0] * 3), int(nxny[1] * 2))

    imgRSx = tike.view.resize_complex_image(phi, scale_factor=(3, 2))
    assert (imgRSx.shape[0] == final_size_xy[1])
    assert (imgRSx.shape[1] == final_size_xy[0])
    assert (imgRSx.dtype == phi.dtype)


###################################################################################################


def test_complexHSV_to_RGB_color_wheel_color_rectangle():

    #============================================================================
    # create color wheel over full +/- pi for the "ground truth" comparison below
    #============================================================================

    nxny = (512, 511)
    colorwheel, H, V = create_color_wheel(nxny)

    # convert color wheel to complex representation, then convert back to HSV and then to RGB:
    colorwheel2 = V * np.exp(1j * H)
    colorwheel2 = tike.view.complexHSV_to_RGB(colorwheel2)

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(colorwheel2[:, :, 0] - colorwheel[:, :, 0])
    axs[0].title.set_text('Red Error')
    axs[1].imshow(colorwheel2[:, :, 1] - colorwheel[:, :, 1])
    axs[1].title.set_text('Green Error')
    axs[2].imshow(colorwheel2[:, :, 2] - colorwheel[:, :, 2])
    axs[2].title.set_text('Blue Error')

    # Error to third decimal place is good enough for by-eye visual inspection purposes
    np.testing.assert_array_almost_equal(colorwheel2, colorwheel, decimal=3)

    #====================================================================================
    # create color wheel over smaller phase range for the "ground truth" comparison below
    #====================================================================================

    nxny = (512, 511)
    colorwheel, H, V = create_color_wheel(nxny, -np.pi / 3.0, +np.pi / 2.0)

    # convert color wheel to complex representation, then convert back to HSV and then to RGB:
    colorwheel2 = V * np.exp(1j * H)
    colorwheel2 = tike.view.complexHSV_to_RGB(colorwheel2)

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(colorwheel2[:, :, 0] - colorwheel[:, :, 0])
    axs[0].title.set_text('Red Error')
    axs[1].imshow(colorwheel2[:, :, 1] - colorwheel[:, :, 1])
    axs[1].title.set_text('Green Error')
    axs[2].imshow(colorwheel2[:, :, 2] - colorwheel[:, :, 2])
    axs[2].title.set_text('Blue Error')

    # Error to third decimal place is good enough for by-eye visual inspection purposes
    np.testing.assert_array_almost_equal(colorwheel2, colorwheel, decimal=3)

    #================================================================================
    # create color rectangle over full +/- pi for the "ground truth" comparison below
    #================================================================================

    sxsy = (201, 101)
    color_rectangle, phs, mag = create_color_rectangle(sxsy)

    # convert color rectangle to complex representation, then convert back to HSV and then to RGB:
    color_rectangle2 = mag * np.exp(1j * phs)
    color_rectangle2 = tike.view.complexHSV_to_RGB(color_rectangle2)

    # plt.figure()
    # plt.imshow( color_rectangle )

    # plt.figure()
    # plt.imshow( color_rectangle2 )

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(color_rectangle2[:, :, 0] - color_rectangle[:, :, 0])
    axs[0].title.set_text('Red Error')
    axs[1].imshow(color_rectangle2[:, :, 1] - color_rectangle[:, :, 1])
    axs[1].title.set_text('Green Error')
    axs[2].imshow(color_rectangle2[:, :, 2] - color_rectangle[:, :, 2])
    axs[2].title.set_text('Blue Error')

    np.testing.assert_array_almost_equal(color_rectangle2,
                                         color_rectangle,
                                         decimal=3)

    #========================================================================================
    # create color rectangle over smaller phase range for the "ground truth" comparison below
    #========================================================================================

    sxsy = (201, 101)
    color_rectangle, phs, mag = create_color_rectangle(sxsy, -np.pi / 3,
                                                       +np.pi / 2)

    # convert color rectangle to complex representation, then convert back to HSV and then to RGB:
    color_rectangle2 = mag * np.exp(1j * phs)
    color_rectangle2 = tike.view.complexHSV_to_RGB(color_rectangle2)

    # plt.figure()
    # plt.imshow( color_rectangle )

    # plt.figure()
    # plt.imshow( color_rectangle2 )

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(color_rectangle2[:, :, 0] - color_rectangle[:, :, 0])
    axs[0].title.set_text('Red Error')
    axs[1].imshow(color_rectangle2[:, :, 1] - color_rectangle[:, :, 1])
    axs[1].title.set_text('Green Error')
    axs[2].imshow(color_rectangle2[:, :, 2] - color_rectangle[:, :, 2])
    axs[2].title.set_text('Blue Error')

    np.testing.assert_array_almost_equal(color_rectangle2,
                                         color_rectangle,
                                         decimal=3)


###################################################################################################

if __name__ == '__main__':

    test_resize_complex_image()
    test_complexHSV_to_RGB_color_wheel_color_rectangle()

    #================================
    # test if complex array is not 2D
    #================================

    sz = (4, 7, 6)
    phi = (2 * np.random.rand(*sz).astype(np.float32) -
           1) + 1j * (2 * np.random.rand(*sz).astype(np.float32) - 1)

    phi_RGB = tike.view.complexHSV_to_RGB(phi)

    assert (phi_RGB.shape[0] == sz[0])
    assert (phi_RGB.shape[1] == sz[1])
    assert (phi_RGB.shape[2] == sz[2])
    assert (phi_RGB.shape[3] == 3)

    plt.figure()
    plt.imshow(phi_RGB[:, :, 2, :])
    plt.show()
