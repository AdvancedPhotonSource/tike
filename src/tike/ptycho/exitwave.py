"""
TODO: fill this part in describing what's happening here
"""

from __future__ import annotations
import dataclasses
import logging
import typing

import cupy as cp
import cupyx.scipy.ndimage
import numpy as np
import numpy.typing as npt

import tike.linalg
import tike.random

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class ExitWaveOptions:
    """Manage data and setting related to exitwave updates."""

    noise_model : str = "gaussian"                      # { "gaussian", "poisson" }
    
    step_length_weight        : float = 0.5             # 0 <= step_length_weight <= 1
    step_length_usemodes      : str = "all_modes"       # { "dominant_mode", "all_modes" }
    step_length_start         : float = 0.5

    unmeasured_pixels_scaling : float = 0.95            
    unmeasured_pixels         : np.array = np.zeros( 1, dtype = float )
    measured_pixels           : np.array = np.ones( 1, dtype = float )

    def copy_to_device(self, comm):
        """Copy to the current GPU memory."""
        if self.unmeasured_pixels is not None:
            self.unmeasured_pixels = cp.asarray( self.unmeasured_pixels )
        if self.measured_pixels is not None:
            self.measured_pixels = cp.asarray( self.measured_pixels )  
        # if self.unmeasured_pixels is not None:
        #     self.unmeasured_pixels = comm.pool.bcast([self.unmeasured_pixels])
        return self

    def copy_to_host(self):
        """Copy to the host CPU memory."""
        if self.unmeasured_pixels is not None:
            self.unmeasured_pixels = cp.asnumpy(self.unmeasured_pixels)
        if self.measured_pixels is not None:
            self.measured_pixels = cp.asnumpy(self.measured_pixels)
        # if self.unmeasured_pixels is not None:
        #     self.unmeasured_pixels = cp.asnumpy( self.unmeasured_pixels[0] )
        return self

    def resample(self, factor: float) -> ExitWaveOptions:
        """Return a new `ExitWaveOptions` with the parameters rescaled."""
        return ExitWaveOptions(
            noise_model               = self.noise_model,
            step_length_weight        = self.step_length_weight,
            step_length_start         = self.step_length_start,
            step_length_usemodes      = self.step_length_usemodes,
            unmeasured_pixels         = self.unmeasured_pixels,
            measured_pixels           = self.measured_pixels, 
            unmeasured_pixels_scaling = self.unmeasured_pixels_scaling
        )
        
###################################################################################################

def poisson_steplength_approx( xi, 
                               abs2_Psi, 
                               I_e, 
                               I_m, 
                               measured_pixels, 
                               step_length, 
                               weight_avg ):

    if measured_pixels.size == 0:
        measured_pixels = 1

    xi_abs_Psi2 = xi * abs2_Psi     
        
    for _ in cp.arange( 0, 2 ):

        xi_alpha_minus_one = xi * step_length[ ..., None, None ] - 1    

        numer = I_m * xi_alpha_minus_one
        denom = abs2_Psi * cp.square( xi_alpha_minus_one ) + I_e - abs2_Psi
        numer = cp.sum( measured_pixels * xi_abs_Psi2 * (1 + numer / denom ), axis = (-1,-2) )

        denom = cp.sum( measured_pixels * cp.square( xi ) * abs2_Psi, axis = (-1,-2) )

        step_length = step_length * ( 1 - weight_avg ) + ( numer / denom ) * weight_avg 

    return step_length

###################################################################################################

def poisson_steplength_ptychoshelves( xi, 
                                      I_e, 
                                      I_m, 
                                      measured_pixels, 
                                      step_length, 
                                      weight_avg ):

    if measured_pixels.size == 0:
        measured_pixels = 1

    denom = measured_pixels * I_e * cp.square( xi )

    sum_denom = cp.sum( denom, axis = ( -1, -2 ))

    for _ in cp.arange( 0, 2 ):

        nom = measured_pixels * xi * ( I_e - I_m / ( 1 - step_length[ ..., None, None ] * xi ) )

        nom_over_denom = cp.sum( nom, axis = ( -1, -2 )) / sum_denom
  
        step_length = ( 1 - weight_avg ) * step_length + weight_avg * nom_over_denom

        step_length = cp.abs( cp.fmax( cp.fmin( step_length, 1 ), 0 ))

    step_length = step_length + cp.random.randn( *step_length.shape ) * 1e-2

    return step_length

###################################################################################################

# def poisson_steplength_exact( xi, 
#                               abs2_Psi, 
#                               I_e, 
#                               I_m, 
#                               measured_pixels, 
#                               step_length_test ):
                              
#     if measured_pixels.size == 0:
#         measured_pixels = 1

#     xi_abs_Psi2        = measured_pixels * xi * abs2_Psi               
#     xi_alpha_minus_one = xi * step_length_test[ ..., None, None, None ] - 1   # cp.newaxis vs None ???   try:  cp.newaxis is None

#     lhs = cp.sum( xi_abs_Psi2 * xi_alpha_minus_one, axis = (-1, -2 ) )  

#     numer = I_m * xi_alpha_minus_one
#     denom = abs2_Psi * cp.square( xi_alpha_minus_one ) + I_e - abs2_Psi

#     rhs = cp.sum( xi_abs_Psi2 * ( numer / denom ), axis = (-1, -2 ) )   

#     f_eq_0 = cp.abs( lhs - rhs )

#     II = cp.argmin( f_eq_0, axis = 0 )

#     step_length = cp.zeros( ( abs2_Psi.shape[0], abs2_Psi.shape[1] ), dtype = 'float32' )      

#     for pp in cp.arange(0, Nscpm ):
#         step_length[ pp, : ] = step_length_test[ II[ pp, : ], pp ]

#     return step_length, f_eq_0
  
###################################################################################################

# if __name__ == "__main__":

    # import h5py

    # f = h5py.File( '/net/s8iddata/export/8-id-ECA/Analysis/atripath/Python/poisson_exwv/pythonTIKE_poisson_testing_31May2023.mat', 'r' )

    # Nspos = int( np.ndarray.item( np.array( f['Nspos'] )))    
    # Nscpm = int( np.ndarray.item( np.array( f['Nscpm'] ))) 

    # sz_Nr = 256
    # sz_Nc = 384

    # xi       = cp.array( f['xi'] ).reshape( ( Nspos, sz_Nr, sz_Nc ), order='F')           # = 1 - I_m / I_e
    # abs2_Psi = cp.array( f['abs2_Psi'] ).reshape( ( Nscpm, Nspos, sz_Nr, sz_Nc ), order='F')
    # I_e      = cp.array( f['I_e'] ).reshape( ( Nspos, sz_Nr, sz_Nc ), order='F')
    # I_m      = cp.array( f['I_m'] ).reshape( ( Nspos, sz_Nr, sz_Nc ), order='F')

    # measured_pixels = cp.array( f['not_I_m_eq0'] ).reshape( ( sz_Nr, sz_Nc ), order='F')
    # alpha_test      = cp.squeeze( cp.array( f['alpha_test'] ))
 
    # #=====

    # poiss_steplength_exact, f_eq_0 = poisson_steplength_exact( xi, abs2_Psi, I_e, I_m, measured_pixels, alpha_test )
    # poiss_steplength_approx        = poisson_steplength_approx( xi, abs2_Psi, I_e, I_m, measured_pixels, 0.50 * cp.ones( ( Nscpm, Nspos )), 0.70 )
    # poiss_steplength_ptychoshelves = poisson_steplength_ptychoshelves( xi, I_e, I_m, measured_pixels, 0.5 * cp.ones( Nspos ), 0.50 )

    # #=====

    # poiss_steplength_exact         = cp.asnumpy( poiss_steplength_exact )
    # poiss_steplength_approx        = cp.asnumpy( poiss_steplength_approx )
    # poiss_steplength_ptychoshelves = cp.asnumpy( poiss_steplength_ptychoshelves )
    # alpha_test                     = cp.asnumpy( alpha_test )
    # f_eq_0                         = cp.asnumpy( f_eq_0 )

    # import matplotlib.pyplot as plt

    # for pp in np.arange( 0, Nscpm ):
    #     plt.subplot( 1, Nscpm, pp + 1 )
    #     ax = plt.gca()
    #     ax.plot( poiss_steplength_exact[ pp, : ], 0.5 + np.arange(0,Nspos), marker='x', color="red")
    #     ax.plot(  np.squeeze( poiss_steplength_ptychoshelves ), 0.5 + np.arange(0,Nspos), marker='.', color="blue")
    #     ax.plot(  poiss_steplength_approx[ pp, : ], 0.5 + np.arange(0,Nspos), marker='.', color="white")
    #     ax.imshow( np.log10(f_eq_0[ :, pp, : ].T), interpolation='none', extent = [ np.min( alpha_test[:,pp] ), np.max( alpha_test[:,pp] ), Nspos, 0 ], aspect="auto" )
        
    # plt.show( block = False )
