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

    x = np.linspace(-nx * 0.5, nx * 0.5, nx)
    y = np.linspace(-ny * 0.5, ny * 0.5, ny)

    xv, yv = np.meshgrid(x, y)

    phi_abs = np.exp(-(xv**2 / (2 * 10 * nx) + yv**2 / (2 * 3 * ny)))
    phi_phs = np.exp(1j * 2 * np.pi * (xv**2 / (2 * 20 * nx) + yv**2 /
                                       (2 * 3 * ny)))

    phi = phi_phs * phi_abs

    return phi


def test_resize_complex_image():

    nxny = (31, 40)

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

    final_size_xy = (int(nxny[0] * resize_factor_xy[0]),
                     int(nxny[1] * resize_factor_xy[1]))

    for x in cv_interp:
        imgRSx = tike.view.resize_complex_image(phi, resize_factor_xy, x)
        assert (imgRSx.shape[0] == final_size_xy[1])
        assert (imgRSx.shape[1] == final_size_xy[0])
        assert (imgRSx.dtype == np.complex128)
        imgRS.append(imgRSx)

    #===============================================
    # resize phi using function default combinations

    imgRSx = tike.view.resize_complex_image(phi)
    assert (imgRSx.shape[0] == nxny[1])
    assert (imgRSx.shape[1] == nxny[0])
    assert (imgRSx.dtype == np.complex128)
    imgRS.append(imgRSx)

    imgRSx = tike.view.resize_complex_image(phi, interpolation=cv.INTER_CUBIC)
    assert (imgRSx.shape[0] == nxny[1])
    assert (imgRSx.shape[1] == nxny[0])
    assert (imgRSx.dtype == np.complex128)
    imgRS.append(imgRSx)

    final_size_xy = (int(nxny[0] * 3), int(nxny[1] * 2))

    imgRSx = tike.view.resize_complex_image(phi, scale_factor=(3, 2))
    assert (imgRSx.shape[0] == final_size_xy[1])
    assert (imgRSx.shape[1] == final_size_xy[0])
    assert (imgRSx.dtype == np.complex128)
    imgRS.append(imgRSx)

    return imgRS


def test_complexHSV_to_RGB():

    nxny = (512, 384)

    #=================================================================================
    # create a complex valued array with somewhat realistic x-ray beam phase structure
    phi = create_testing_complex_wavefield(nxny)

    # phi = phi * 0
    # phi = np.array([ 0 + 0j,])

    #============================================================================================================================
    # represent this complex valued array in a HSV representation with (H)ue as phase, (V)alue as magnitude, and (S)aturation = 1
    rgb_imgRS = tike.view.complexHSV_to_RGB(phi)

    result_dir = os.path.join(os.path.dirname(__file__), 'result')
    os.makedirs(result_dir, exist_ok=True)
    plt.imsave(
        os.path.join(result_dir, 'hsv_complex.png'),
        rgb_imgRS,
    )

    # assert(np.isreal(rgb_img))
    # np.testing.assert_equal

    return rgb_imgRS


def test_complexHSV_simple_inputs():

    #====================================
    # test if giving single complex zero:

    result = tike.view.complexHSV_to_RGB(np.array([
        0 + 0j,
    ]))

    the_answer = np.array([[0., 0., 0.]], dtype='float32')

    np.testing.assert_array_equal(result, the_answer)

    #==================================
    # test if giving array of all zeros:

    A = np.zeros((10, 11), 'float32')

    result = tike.view.complexHSV_to_RGB(A)

    the_answer = np.zeros((10, 11, 3), 'float32')

    np.testing.assert_array_equal(result, the_answer)

    #================================================================
    # test with respect to a simple constant array of ones and zeros:

    A = np.array([ [0, 0, 0],
                   [1, 1, 1],
                   [1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1], ]) # yapf:disable

    the_answer = np.array([[[0.       , 0.       , 0.       ],
                            [0.       , 0.       , 0.       ],
                            [0.       , 0.       , 0.       ]],
                           [[0.9999999, 0.       , 0.       ],
                            [0.9999999, 0.       , 0.       ],
                            [0.9999999, 0.       , 0.       ]],
                           [[0.9999999, 0.       , 0.       ],
                            [0.       , 0.       , 0.       ],
                            [0.       , 0.       , 0.       ]],
                           [[0.       , 0.       , 0.       ],
                            [0.9999999, 0.       , 0.       ],
                            [0.       , 0.       , 0.       ]],
                           [[0.       , 0.       , 0.       ],
                            [0.       , 0.       , 0.       ],
                            [0.9999999, 0.       , 0.       ]]], dtype='float32') # yapf:disable

    result = tike.view.complexHSV_to_RGB(A)

    np.testing.assert_array_almost_equal(result, the_answer, decimal=7)


if __name__ == '__main__':

    test_complexHSV_simple_inputs()

    imgRS = test_resize_complex_image()
    rgb_imgRS = test_complexHSV_to_RGB()
