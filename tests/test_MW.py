# test multi_energy feature

import tike.ptycho
from tike.ptycho.probe import add_modes_random_phase
from tike.ptycho.probe_MW import MW_probe

import matplotlib.pyplot as plt
import h5py
import numpy as np
import cupy as cp
import os


def gen_position(steppix_x, steppix_y, N_scan_x, N_scan_y, probe_shape):
    # generate scan position for ptychography
    ppx =np.arange(-np.floor(N_scan_x/2.0),np.ceil(N_scan_x/2.0))*steppix_x #x direction, column
    ppy =np.arange(-np.floor(N_scan_y/2.0),np.ceil(N_scan_y/2.0))*steppix_y #x row, row

    [ppX, ppY] = np.meshgrid(ppx,ppy)
    ppX = np.reshape(ppX, (np.product(ppX.shape),1))
    ppY = np.reshape(ppY, (np.product(ppY.shape),1))
    
    ppX = ppX - np.min(ppX) + probe_shape
    ppY = ppY - np.min(ppY) + probe_shape

    scan = np.hstack((ppX, ppY)) 
    scan = scan[np.newaxis]
    return scan
    
def create_dataset(testdir, energy, nmodes):
    dis_StoD = 2
    dis_defocus = 500e-6
    detector_shape = 128
    detector_pixel = 75e-6
    # generate probe
    probe,_ = MW_probe(detector_shape,energy,detector_pixel,
        dis_defocus,dis_StoD)

    # add nmodes to probe
    probe = add_modes_random_phase(probe, nmodes)

    # read simulated object
    amplitude = plt.imread(os.path.join(
        testdir, "data/baboon512.png"))
    phase = plt.imread(os.path.join(
        testdir, "data/lena512.png"))
    obj = amplitude*np.exp(1j*phase* np.pi)
    obj = obj[np.newaxis]

    #generate ptychography scan point
    N_scan = 11 #scan point in one direction
    lam = 1.24e-9/8.8 # central wavelength
    # object plane pixel size
    dx = lam*dis_StoD/detector_shape/detector_pixel 
    step = 300e-9/dx #step size in pixel
    scan = gen_position(step, step, N_scan, N_scan,detector_shape)

    # test ptycho.simulate
    data = tike.ptycho.simulate(
            detector_shape,
            probe, scan,
            obj,
            )

    return data,obj,probe,scan



if __name__ == "__main__":

    testdir = '/home/beams/YUDONGYAO/code/Tike/tike/tests'
    energy = 10
    nmodes = 5

    #generate simulated dataset
    data,obj,probe,scan = create_dataset(testdir,energy,nmodes)

    # ptycho reconstruction
    result = tike.ptycho.reconstruct(
                data=data,
                probe = probe,
                scan = scan,
                algorithm='combined',
                num_iter=10,
                nmode=nmodes,
                energy = energy,
                recover_psi=True,
                recover_probe=True,
                recover_positions=True,
                rtol=-1,
                model='poisson',
            )       
