"""
This is a preliminary forward/inverse

The purpose of this test is to demonstrate the numerical instability that arises 
when we try to update a 3D sample via an update scheme where a 2D slice is updated
then that result is propagated to the next slice, and this is repeated until we 
reach the first slice.

"""

import cupy as cp
import numpy as np
import scipy
import pathlib
# import time

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# import sys
import os

# import mat73
import h5py

import tike.view

#==================================================================================================
# load and define the 3D sample transmission

def load_3Dsample_from_mat( input_file: pathlib.Path ):

    f = h5py.File( input_file )

    Zre = f[ 'Tvol3D' ][ ... ][ 'real' ]
    Zim = f[ 'Tvol3D' ][ ... ][ 'imag' ]
    Tvol3D = np.transpose( np.asarray( Zre + 1j * Zim, dtype = np.csingle ))

    Tvol3D = Tvol3D[ :, :, 0 : 150 ] 

    #####

    # proj_dim0_Tvol3D = np.prod( Tvol3D, axis = 0 ) #/ Tvol3D.shape[ 0 ]
    # proj_dim1_Tvol3D = np.prod( Tvol3D, axis = 1 ) #/ Tvol3D.shape[ 1 ]
    # proj_dim2_Tvol3D = np.prod( Tvol3D, axis = 2 ) #/ Tvol3D.shape[ 2 ]

    # from mpl_toolkits.axes_grid1 import make_axes_locatable

    # fig, ax = plt.subplots( nrows = 1, ncols = 3, )

    # pos0 = ax[0].imshow( np.abs( proj_dim0_Tvol3D ), cmap = 'turbo', )
    # ax[0].set_title('sum0, proj along y (bird''s eye)')
    # ax[0].set_xlabel("z-axis")
    # ax[0].set_ylabel("x-axis")
    # divider1 = make_axes_locatable(ax[0])
    # cax0 = divider1.append_axes("right", size="5%", pad=0.1,)
    # fig.colorbar(pos0, cax=cax0, fraction=0.046, pad=0.04)

    # pos1 = ax[1].imshow( np.abs( proj_dim1_Tvol3D ), cmap = 'turbo', )
    # ax[1].set_title('sum1, proj along x (side view)')
    # ax[1].set_xlabel("z-axis")
    # ax[1].set_ylabel("y-axis")
    # divider1 = make_axes_locatable(ax[1])
    # cax1 = divider1.append_axes("right", size="5%", pad=0.1,)
    # fig.colorbar(pos1, cax=cax1, fraction=0.046, pad=0.04)

    # pos2 = ax[2].imshow( np.abs( proj_dim2_Tvol3D ), cmap = 'turbo', )
    # ax[2].set_title('sum2, proj along z (beam)')
    # ax[2].set_xlabel("x-axis")
    # ax[2].set_ylabel("y-axis")
    # divider2 = make_axes_locatable(ax[2])
    # cax2 = divider2.append_axes("right", size="5%", pad=0.1,)
    # fig.colorbar(pos2, cax=cax2, fraction=0.046, pad=0.04 )

    # fig.tight_layout()
    # plt.show(block=False)

    # plt.close( fig )

    #####

    return Tvol3D

#==================================================================================================
# define probe function

def create_2Dprobe_blurred_circ( array_size, circ_radius, blur_gauss_stdev ):

    center = ( int( array_size[0] / 2 ), 
               int( array_size[1] / 2 ))

    X, Y = np.meshgrid( np.linspace( 0, array_size[1], array_size[1] ), 
                        np.linspace( 0, array_size[0], array_size[0] ))

    ellipse = ( (((( X - center[1]) / circ_radius[1] ) ) ** 2 ) + (((( Y - center[0]) / circ_radius[0] ) ) ** 2 )) <= 1
    
    ellipse_blurred = scipy.ndimage.gaussian_filter( input = np.abs( np.asarray( ellipse, dtype='cfloat')), 
                                                     sigma = blur_gauss_stdev )
    
    return ellipse_blurred

#==================================================================================================
# define Fresnel integral spectrum propagation parameters

def create_spectrum_propagator( N, wavelength, delta_z, prb_FOV ):

    rr2 = np.linspace( -0.5 * N[0], 0.5 * N[0] - 1, num = N[0] ) ** 2
    cc2 = np.linspace( -0.5 * N[1], 0.5 * N[1] - 1, num = N[1] ) ** 2

    x = -1j * np.pi * wavelength * delta_z
    rr2 = np.exp( x * rr2[ ..., None ] / ( prb_FOV[0] ** 2 ))
    cc2 = np.exp( x * cc2[ ..., None ] / ( prb_FOV[1] ** 2 ))

    spect_prop_phase_curvature = np.fft.fftshift( np.outer( np.transpose( rr2 ), cc2 ))

    return spect_prop_phase_curvature

#==================================================================================================

def forward_multislice( obj3D, phi, spect_prop_phase_curvature, ):

    exwv3D = cp.zeros( obj3D.shape, dtype = obj3D.dtype )
    prb3D  = cp.zeros( obj3D.shape, dtype = obj3D.dtype )

    for tt in cp.arange( 0, obj3D.shape[-1], 1 ) :

        prb3D[  :, :, tt ] = phi
        exwv3D[ :, :, tt ] = obj3D[ :, :, tt ] * phi

        if tt == obj3D.shape[-1]:
            break

        phi = cp.fft.ifft2( cp.fft.fft2( exwv3D[ :, :, tt ] ) * spect_prop_phase_curvature )

    return exwv3D, prb3D

#==================================================================================================

def inverse_multislice_ePIE_v0( psi, obj3D, prb3D, spect_prop_phase_curvature ):

    spect_prop_phase_curvature = cp.conj( spect_prop_phase_curvature )

    obj3D_backwards  = cp.ones(  obj3D.shape, dtype = obj3D.dtype )
    exwv3D_backwards = cp.zeros( obj3D.shape, dtype = obj3D.dtype )
    prb3D_backwards  = cp.zeros( obj3D.shape, dtype = obj3D.dtype )

    for tt in cp.arange( ( obj3D.shape[-1] - 1 ), -1, -1 ):
        
        dX  = ( psi - obj3D[ :, :, tt ] * prb3D[ :, :, tt ] )
        obj = obj3D[ :, :, tt ] + ( 1 / cp.max( cp.abs( prb3D[ :, :, tt ] ) ** 2 )) * cp.conj( prb3D[ :, :, tt ] ) * dX
        phi = prb3D[ :, :, tt ] + ( 1 / cp.max( cp.abs( obj3D[ :, :, tt ] ) ** 2 )) * cp.conj( obj3D[ :, :, tt ] ) * dX

        exwv3D_backwards[ :, :, tt ] = psi
        obj3D_backwards[  :, :, tt ] = obj
        prb3D_backwards[  :, :, tt ] = phi

        if tt == 0:
            break

        psi = cp.fft.ifft2( cp.fft.fft2( phi ) * spect_prop_phase_curvature )

    return exwv3D_backwards, obj3D_backwards, prb3D_backwards

#==================================================================================================

if  __name__ == "__main__":

    #############
    # setup paths
    #############
        
    os.chdir( os.path.dirname( os.path.realpath( __file__ )))

    output_folder = pathlib.Path( 'result/multislice_3PIE_numerical_stability/' )

    output_folder.mkdir( parents = True, exist_ok = True)
                                
    ###############################
    # create 3D sample and 2D probe
    ###############################

    Tvol3D = load_3Dsample_from_mat( pathlib.Path( 'data/Tvol3D_2cubes_2spheres_r256xc384xp512.mat' ))

    NrNc = np.asarray( Tvol3D.shape[0:-1], dtype = np.uint32 )  # in this case, the transverse sample and probe arrays have the same size

    circ_radius      = np.asarray( [ 0.30 * NrNc[ 0 ], 0.30 * NrNc[ 1 ] ], dtype = np.float32 ) * 0.5
    blur_gauss_stdev = np.asarray( [ 0.02 * NrNc[ 0 ], 0.02 * NrNc[ 1 ] ], dtype = np.float32 )

    probe2D = create_2Dprobe_blurred_circ( NrNc, circ_radius, blur_gauss_stdev )

    #################################
    # spectrum propagation parameters
    #################################

    energy     = 10.0                           # x-ray energy ( keV )    
    wavelength = ( 12.4 / energy ) * 1e-10      # wavelength ( meters )
    
    csys_z_prb_Ly = NrNc[0] * 10.0e-9           # total transverse field-of-view of probe array ( meters )
    csys_z_prb_Lx = NrNc[1] * 10.0e-9

    delta_z = 50.0e-9                           # multislice spectrum propagation distance ( BE CAREFUL WITH THIS, OR ALIASING OCCURS )

    prb_FOV = np.asarray( [ csys_z_prb_Ly, csys_z_prb_Lx ], dtype = np.float32 )

    spect_prop_phase_curvature = create_spectrum_propagator( NrNc, wavelength, delta_z, prb_FOV )

    #############
    # move to GPU
    #############

    Tvol3D  = cp.asarray( Tvol3D )
    probe2D = cp.asarray( probe2D )

    Tvol3D_0  = Tvol3D
    probe2D_0 = probe2D
    
    spect_prop_phase_curvature = cp.asarray( spect_prop_phase_curvature )

    ##########################################
    # generate a "measurement" for 3PIE to use 
    ##########################################

    exwv3D, _ = forward_multislice( Tvol3D, probe2D, spect_prop_phase_curvature )

    exwv3D_meas = cp.abs( cp.fft.fft2( exwv3D[  :, :, -1 ] ))

    ############################################################################
    # what does the multislice exitwave vs projection approx exitwave look like?
    ############################################################################

    # X = exwv3D[  :, :, -1 ].get()
    # absX = np.abs( X )
    # X = ( absX / ( 1e-5 + np.max( absX ))) * np.exp( 1j * np.angle( X ))
    # exwv3D_HSV = tike.view.complexHSV_to_RGB( X )

    fig, ax = plt.subplots( nrows = 2, ncols = 2,  num = 101 )

    pos0 = ax[0,0].imshow( np.abs( np.prod( Tvol3D.get(), axis = -1 ) * probe2D.get() ), cmap='turbo' )
    ax[0,0].set_aspect('equal')
    ax[0,0].set_title('abs, proj approx exwv')
    divider0 = make_axes_locatable(ax[0,0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.1,)
    fig.colorbar(pos0, cax=cax0, fraction=0.046, pad=0.04)

    pos1 = ax[0,1].imshow( np.angle( np.prod( Tvol3D.get(), axis = -1 ) * probe2D.get() ), cmap='hsv' )
    ax[0,1].set_aspect('equal')
    ax[0,1].set_title('phs, proj approx exwv')
    divider1 = make_axes_locatable(ax[0,1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1,)
    fig.colorbar(pos1, cax=cax1, fraction=0.046, pad=0.04)

    pos2 = ax[1,0].imshow( np.abs( exwv3D.get()[ ..., -1 ] ), cmap='turbo' )
    ax[1,0].set_aspect('equal')
    ax[1,0].set_title('abs, multislice exwv')
    divider2 = make_axes_locatable(ax[1,0])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1,)
    fig.colorbar(pos2, cax=cax2, fraction=0.046, pad=0.04)

    pos3 = ax[1,1].imshow( np.angle( exwv3D.get()[ ..., -1 ] ), cmap='hsv' )
    ax[1,1].set_aspect('equal')
    ax[1,1].set_title('phs, multislice exwv')
    divider3 = make_axes_locatable(ax[1,1])
    cax3 = divider3.append_axes("right", size="5%", pad=0.1,)
    fig.colorbar(pos3, cax=cax3, fraction=0.046, pad=0.04)
    
    fig.tight_layout()

    #plt.show( block = False )
    plt.savefig( output_folder / 'multislice_exwv_vs_proj_approx_exwv.png', dpi = 200 )

    plt.close('all')

    #########################################################
    # now crank forward and inverse multislice back and forth
    #########################################################

    Nsweeps = 250

    Tvol3D_error  = cp.zeros( Nsweeps, dtype = cp.float32 )
    probe2D_error = cp.zeros( Nsweeps, dtype = cp.float32 )

    for tt in cp.arange( 0, Nsweeps, 1 ):
        
        print( 'forw/back sweep =', tt )

        exwv3D, prb3D = forward_multislice( Tvol3D, probe2D, spect_prop_phase_curvature )
        
        #####################################################################################################
        # propagate exitwave plane to far field plane, apply meas constraint, backpropagate to exitwave plane

        # # !!!!!!!!!! THIS MAKES THINGS WAAAAY WORSE !!!!!!!!!!!!!!!!
        # exwv3D[ :, :, -1 ] = cp.fft.ifft2( exwv3D_meas * cp.exp( 1j * cp.angle( cp.fft.fft2( exwv3D[  :, :, -1 ] ))))

        #####################################################################################################

        exwv3D_backwards, Tvol3D_backwards, prb3D_backwards = inverse_multislice_ePIE_v0( exwv3D[ :, :, -1 ], 
                                                                                          Tvol3D, 
                                                                                          prb3D, 
                                                                                          spect_prop_phase_curvature )
        
        Tvol3D_error[ tt ]  = cp.linalg.norm( Tvol3D_backwards - Tvol3D_0 )
        probe2D_error[ tt ] = cp.linalg.norm( prb3D_backwards[ :, :, 0 ] - probe2D_0 )

        Tvol3D  = Tvol3D_backwards
        probe2D = prb3D_backwards[ :, :, 0 ]

        #####################################################
        # plotting of 3D sample projection vs forw/back sweep
        #####################################################
            
        if (( tt % 50 ) == 0 ) or ( tt == Nsweeps - 1):     

            fig, ax = plt.subplots( nrows = 2, ncols = 2,  num = 101 )

            pos0 = ax[0,0].imshow( np.abs( np.prod( Tvol3D_0.get(), axis = -1 ) ), cmap='turbo' )
            ax[0,0].set_aspect('equal')
            ax[0,0].set_title('abs proj_z ground truth')
            divider0 = make_axes_locatable(ax[0,0])
            cax0 = divider0.append_axes("right", size="5%", pad=0.1,)
            fig.colorbar(pos0, cax=cax0, fraction=0.046, pad=0.04)

            pos1 = ax[0,1].imshow( np.angle( np.prod( Tvol3D_0.get(), axis = -1 ) ), cmap='hsv' )
            ax[0,1].set_aspect('equal')
            ax[0,1].set_title('phs proj_z ground truth')
            divider1 = make_axes_locatable(ax[0,1])
            cax1 = divider1.append_axes("right", size="5%", pad=0.1,)
            fig.colorbar(pos1, cax=cax1, fraction=0.046, pad=0.04)

            pos2 = ax[1,0].imshow( np.abs( np.prod( Tvol3D.get(), axis = -1 ) ), cmap='turbo' )
            ax[1,0].set_aspect('equal')
            ax[1,0].set_title('abs proj_z forw/back multislice')
            divider2 = make_axes_locatable(ax[1,0])
            cax2 = divider2.append_axes("right", size="5%", pad=0.1,)
            fig.colorbar(pos2, cax=cax2, fraction=0.046, pad=0.04)

            pos3 = ax[1,1].imshow( np.angle( np.prod( Tvol3D.get(), axis = -1 ) ), cmap='hsv' )
            ax[1,1].set_aspect('equal')
            ax[1,1].set_title('phs proj_z forw/back multislice')
            divider3 = make_axes_locatable(ax[1,1])
            cax3 = divider3.append_axes("right", size="5%", pad=0.1,)
            fig.colorbar(pos3, cax=cax3, fraction=0.046, pad=0.04)

            fig.tight_layout()
            
            plt.savefig( output_folder / ('multislice_forward_backward_sweep_' + str(tt) + '.png' ), dpi = 300 )

            plt.close( fig )

    #####

    fig, ax = plt.subplots( nrows = 1, ncols = 2,  num = 101 )

    pos0 = ax[0].plot( Tvol3D_error.get() )
    ax[0].grid()
    ax[0].set_title('Tvol3D_error')
    pos1 = ax[1].plot( probe2D_error.get() )
    ax[1].grid()
    ax[1].set_title('probe2D_error')

    # plt.show( block = False )
    plt.savefig( output_folder / ( 'multislice_forward_backward_error_vs_' + str( Nsweeps ) + 'sweep.png' ), dpi = 300 )

    plt.close( fig )


