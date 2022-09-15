#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test functions in tike.trajectory."""

from signal import Handlers
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import cv2 as cv
import tike.view
import os.path

__author__ = "Ash Tripathi, "
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


def create_colorwheel( nxny ):

    #================================================================
    # numerically create the value and hue components of a colorwheel

    nx, ny = nxny

    xx, yy = np.meshgrid( np.linspace( -1, 1, nx ), np.linspace( -1, 1, ny ))

    V = np.sqrt( xx ** 2 + yy ** 2 )    
    V[ V > 1 ] = 0

    H = ( np.arctan2( xx, yy ) + np.pi ) / ( np.pi * 2 )

    #================================================================
    # assign the value and hue componenets to a ny x nx x 3 HSV image

    hsv_img = np.ones( ( ny, nx, 3 ), 'float32' )

    hsv_img[ ..., 0 ] = H 
    hsv_img[ ..., 2 ] = V

    #==================================
    # convert HSV representation to RGB

    rgb_img = mplcolors.hsv_to_rgb( hsv_img )

    return rgb_img, H, V


def create_colorrectangle( sxsy ):

    #====================================================================
    # numerically create the value and hue components of a color rectangle

    szx, szy = sxsy

    mag = np.tile( np.linspace( 0, 1, szx ), ( szy, 1 ) )
    phs = np.linspace( -np.pi, np.pi, szy ).reshape( szy, 1 )
    phs = np.tile( phs, ( 1, szx ))

    #================================================================
    # assign the value and hue componenets to a ny x nx x 3 HSV image

    hsv_img = np.ones( ( szy, szx, 3 ), 'float32' )

    hsv_img[ ..., 0 ] = phs 
    hsv_img[ ..., 2 ] = mag

    #================================
    # Rescale hue to the range [0, 1]

    eps = np.finfo( np.float32 ).eps

    hsv_img[ ..., 0 ] -= np.min( hsv_img[ ..., 0 ] )
    hsv_img[ ..., 0 ] = hsv_img[ ..., 0 ] / ( eps + np.max( hsv_img[ ..., 0 ] ))

    #==================================
    # convert HSV representation to RGB

    color_rectangle = mplcolors.hsv_to_rgb( hsv_img )

    return color_rectangle, phs, mag



def test_resize_complex_image():

    nxny = (31, 40)

    resize_factor_xy = (2.432, 3.867)

    cv_interp = [
        cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_AREA,
        cv.INTER_LANCZOS4
    ]

    #=================================================================================
    # create a complex valued array with somewhat realistic x-ray beam phase structure

    phi = ( 2 * np.random.rand( nxny[1], nxny[0] ).astype( np.float32 ) - 1 ) + 1j * ( 2 * np.random.rand( nxny[1], nxny[0] ).astype( np.float32 ) - 1 )

    #=================================================================
    # resize phi using different built in openCV interpolation methods

    final_size_xy = (int(nxny[0] * resize_factor_xy[0]),
                     int(nxny[1] * resize_factor_xy[1]))

    for x in cv_interp:
        imgRSx = tike.view.resize_complex_image(phi, resize_factor_xy, x)
        assert( imgRSx.shape[0] == final_size_xy[1])
        assert( imgRSx.shape[1] == final_size_xy[0])
        assert( imgRSx.dtype == phi.dtype )

    #===============================================
    # resize phi using function default combinations

    imgRSx = tike.view.resize_complex_image(phi)
    assert( imgRSx.shape[0] == nxny[1])
    assert( imgRSx.shape[1] == nxny[0])
    assert( imgRSx.dtype == phi.dtype )

    imgRSx = tike.view.resize_complex_image(phi, interpolation=cv.INTER_CUBIC)
    assert( imgRSx.shape[0] == nxny[1])
    assert( imgRSx.shape[1] == nxny[0])
    assert( imgRSx.dtype == phi.dtype )

    final_size_xy = ( int( nxny[0] * 3 ), 
                      int( nxny[1] * 2 ) )

    imgRSx = tike.view.resize_complex_image(phi, scale_factor=(3, 2))
    assert( imgRSx.shape[0] == final_size_xy[1])
    assert( imgRSx.shape[1] == final_size_xy[0])
    assert( imgRSx.dtype == phi.dtype )
    
def test_complexHSV_to_RGB_colorwheel_colorrectangle():

    #==========================================================
    # create colorwheel for the "ground truth" comparison below

    nxny = ( 512, 511 ) 
    colorwheel, H, V = create_colorwheel( nxny )

    # convert color wheel to complex representation, then convert back to HSV and then to RGB:
    colorwheel2 = V * np.exp( 1j * H + 0 * 1j * np.pi / 1 )
    colorwheel2 = tike.view.complexHSV_to_RGB( colorwheel2 )

    # plt.figure()
    # plt.imshow( colorwheel )
    # plt.show(block=False)

    # plt.figure()
    # plt.imshow( colorwheel2 )
    # plt.show(block=False)

    # fig, axs = plt.subplots(1, 3)
    # axs[0].imshow( colorwheel2[:,:,0] - colorwheel[:,:,0] )
    # axs[0].title.set_text('Red Error')
    # axs[1].imshow( colorwheel2[:,:,1] - colorwheel[:,:,1] )
    # axs[1].title.set_text('Green Error')
    # axs[2].imshow( colorwheel2[:,:,2] - colorwheel[:,:,2] )
    # axs[2].title.set_text('Blue Error')
    # plt.show(block=False)

    # np.sum( colorwheel2 - colorwheel ) / np.sum( colorwheel )

    # Error to second decimal place is good enough for by-eye visual inspection purposes
    np.testing.assert_array_almost_equal( colorwheel2, colorwheel, decimal = 2 )

    #===============================================================
    # create color rectangle for the "ground truth" comparison below

    sxsy = ( 201, 101 ) 
    color_rectangle, phs, mag = create_colorrectangle( sxsy )

    # convert color rectangle to complex representation, then convert back to HSV and then to RGB:
    color_rectangle2 = mag * np.exp( 1j * phs )
    color_rectangle2 = tike.view.complexHSV_to_RGB( color_rectangle2 )

    # plt.figure()
    # plt.imshow( color_rectangle )
    # plt.show(block=False)

    # plt.figure()
    # plt.imshow( color_rectangle2 )
    # plt.show(block=False)

    np.testing.assert_array_equal( color_rectangle2, color_rectangle )

#==================================================================================================
#==================================================================================================

if __name__ == '__main__':

    test_resize_complex_image()
    test_complexHSV_to_RGB_colorwheel_colorrectangle()

