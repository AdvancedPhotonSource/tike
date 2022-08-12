#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test functions in tike.trajectory."""

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import tike.view
import os.path

__author__ = "Ash Tripathi, "
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


def create_testing_complex_wavefield(nxny):

    nx, ny = nxny

    x = np.linspace(-nx, nx, 2 * nx)
    y = np.linspace(-ny, ny, 2 * ny)

    xv, yv = np.meshgrid(x, y)

    phi_abs = np.exp(-(xv**2 / (2 * 10 * nx) + yv**2 / (2 * 3 * ny)))
    phi_phs = np.exp(1j * 2 * np.pi * (xv**2 / (2 * 20 * nx) + yv**2 /
                                       (2 * 3 * ny)))

    phi = phi_phs * phi_abs

    return phi


def test_resize_complex_image():

    nxny = (32, 40)

    imgRS = []

    resize_factor_xy = (2.432, 3.867)

    cv_interp = [
        cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_AREA,
        cv.INTER_LANCZOS4
    ]

    #=================================================================================
    # create a complex valued array with somewhat realistic x-ray beam phase structure

    phi = create_testing_complex_wavefield(nxny)

    #=================================================================
    # resize phi using different built in openCV interpolation methods

    for x in cv_interp:
        imgRSx = tike.view.resize_complex_image(phi, resize_factor_xy, x)
        imgRS.append(imgRSx)

    #===============================================
    # resize phi using function default combinations

    imgRSx = tike.view.resize_complex_image(phi, interpolation=cv.INTER_CUBIC)
    imgRS.append(imgRSx)

    imgRSx = tike.view.resize_complex_image(phi, scale_factor=(3, 2))
    imgRS.append(imgRSx)

    imgRSx = tike.view.resize_complex_image(phi)
    imgRS.append(imgRSx)

    return imgRS


def test_complexHSV_to_RGB():

    nxny = (512, 384)

    #=================================================================================
    # create a complex valued array with somewhat realistic x-ray beam phase structure
    phi = create_testing_complex_wavefield(nxny)

    #============================================================================================================================
    # represent this complex valued array in a HSV representation with (H)ue as phase, (V)alue as magnitude, and (S)aturation = 1
    rgb_imgRS = tike.view.complexHSV_to_RGB(phi)

    result_dir = os.path.join(os.path.dirname(__file__), 'result')
    os.makedirs(result_dir, exist_ok=True)
    plt.imsave(
        os.path.join(result_dir, 'hsv_complex.png'),
        rgb_imgRS,
    )

    return rgb_imgRS


def test_complexHSV_simple_inputs():

    # FIXME: Function fails when all inputs are zero
    result = tike.view.complexHSV_to_RGB(np.array([
        0 + 0j,
        # FIXME: Add more input output pairs to this test
    ]))
    np.assert_equal(result, np.array([
        [0, 0, 0],
        # For example, it should be trivial to figure out which inputs generate
        # these outputs (or similar):
        # [1, 1, 1],
        # [1, 0, 0],
        # [0, 1, 0],
        # [0, 0, 1],
    ]))


if __name__ == '__main__':

    imgRS = test_resize_complex_image()
    rgb_imgRS = test_complexHSV_to_RGB()
