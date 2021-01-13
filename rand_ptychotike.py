import importlib
import logging

import matplotlib.pyplot as plt
import numpy as np
import cupy 
import tike
import tike.ptycho
import tike.view
import dxchange
import cv2
import sys

for module in [tike, np]:
    print("{} is version {}".format(module.__name__, module.__version__))


if __name__ == "__main__":
    if (len(sys.argv)<2):
        igpu = 0
    else:
        igpu = np.int(sys.argv[1])

    pool = cupy.cuda.MemoryPool(cupy.cuda.malloc_managed)
    cupy.cuda.set_allocator(pool.malloc)
    with cupy.cuda.Device(igpu):
        #Test data1
        # amplitude = plt.imread("tests/data/Cryptomeria_japonica-0128.png") * 0 + 1
        # phase = plt.imread("tests/data/Bombus_terrestris-0128.png")   #* np.pi 

        #Test data2
        # amplitude = plt.imread("tests/data/Cryptomeria_japonica-0128.png") * 0 + 1
        # _phase = np.float32(np.expand_dims(cv2.resize(plt.imread('tests/data/cameraman-0256.tif'), (128,128), interpolation=cv2.INTER_CUBIC), 0))
        # _min1, _max1 = _phase.min(), _phase.max()
        # phase = (_phase - _min1) / (_max1 - _min1)
        # print(np.min(phase), np.max(phase))
        # phase = phase * np.pi

        #Test data3
        # phase = plt.imread("tests/data/Cryptomeria_japonica-0128.png") 
        # amplitude = plt.imread("tests/data/Bombus_terrestris-0128.png") *0 + 1

        #Test data4
        # amplitude = plt.imread("tests/data/Cryptomeria_japonica-0128.png") * 0 + 1
        # _phase = np.float32(np.expand_dims(cv2.resize(plt.imread("tests/data/lena-0512.tiff"), (128,128), interpolation=cv2.INTER_CUBIC), 0))

        phase = dxchange.read_tiff("tests/data/cone-0512padded.tiff")
        amplitude = np.ones_like(phase)

        # _min1, _max1 = _phase.min(), _phase.max()
        # phase = (_phase - _min1) / (_max1 - _min1)
        print(np.min(phase), np.max(phase))

        ntheta = 1  # number angular views
        original = np.tile(amplitude.copy() * np.exp(1j * phase.copy()), (ntheta, 1, 1)).astype('complex64')

        # plt.subplot(1, 2, 1)
        # plt.imshow(amplitude, vmin=0, vmax=1, cmap=plt.cm.twilight)
        # cb0 = plt.colorbar(orientation='horizontal')
        # plt.subplot(1, 2, 2)
        # plt.imshow(phase, vmin=0, vmax=1, cmap=plt.cm.twilight)
        # cb1 = plt.colorbar(orientation='horizontal')

        # dxchange.write_tiff(np.angle(original[0]), 'results/org_imag_phase')
        # dxchange.write_tiff(np.abs(original[0]), 'results/org_imag_amp')
        # print(original.shape)

        #Define the probe
        # from tike.ptycho.probe import add_modes_random_phase
        # np.random.seed(0)
        pw = 32 # probe width
        nmode = 1
        weights = tike.ptycho.gaussian(pw, rin=0.3, rout=1.0)
        # probe = weights * np.exp(1j * 0) * 10
        name = '7Cone512paddedPoissongrp1noise1prbFscanFiter50test'
        grp = 1 #number of groups to sample
        probe = weights * np.exp(1j * 0.2 * weights) * 1
        # to control noise level *1,5,10 etc --> gets less noisy
        probe = np.tile(probe, (ntheta, 1, 1)).astype('complex64')[:, np.newaxis, np.newaxis, np.newaxis]
        # probe = add_modes_random_phase(probe, nmode)

        #Define the trajectory
        buffer = pw/2
        v, h = np.meshgrid(
            np.linspace(buffer, amplitude.shape[0]-pw-buffer, 120, endpoint=True),
            np.linspace(buffer, amplitude.shape[0]-pw-buffer, 120, endpoint=True),
            indexing='ij'
        )

        scan = np.tile(np.stack((np.ravel(v), np.ravel(h)), axis=1), (ntheta, 1, 1)).astype('float32')
        np.random.seed(100)
        true_scan = scan + (np.random.rand(*scan.shape) - 0.5) * 4

        # true_scan = scan
        print('scan', scan.shape)

        # provide freq information that needs to be used
        xp = cupy
        n = pw * 2
        M = scan.shape[1]
        [_, kv, ku] = xp.mgrid[0:M, -n // 2:n // 2, -n // 2:n // 2] / n
        ku = xp.fft.fftshift(ku, axes=(-1, -2))
        kv = xp.fft.fftshift(kv, axes=(-1, -2))
        ku = ku.reshape(M, -1).astype('float32')
        kv = kv.reshape(M, -1).astype('float32')
        freqs = xp.stack((kv, ku), axis=-1)
        # freq = freq[..., ::7, :]    #choose only 1 out of 7 freq.
        print('stack freqs:', freqs.shape)                            
        print('freqs data dtype', freqs.dtype)
        # dxchange.write_tiff(np.fft.fftshift(np.log(data_all)), 'results/dataRecon/dataall'+name)
        
        # randomly subsample the full frequency space and run ptycho solver
        #target indexes for randomization
        numindexes = freqs.shape[1] // grp
        freq = xp.zeros((grp, freqs.shape[0], numindexes, 2)).astype('float32')
        print('freq shape', freq.shape)
        idx = np.arange(freqs.shape[1], dtype=int)
        for m in range(freqs.shape[0]):
            np.random.shuffle(idx)
            for n in range(0, grp):
                freq[n, m] = freqs[m, idx[n*numindexes:(n+1)*numindexes], :]
                # print('freq shape', freq.shape)

        # print('freq2 shape', freq[2].shape)
        # numindexes = freqs.shape[1] // grp
        # print('each grpsize', numindexes)
        # idx = np.zeros((grp, freqs.shape[0], numindexes)) 
        # freq = np.zeros((grp, freqs.shape[0], numindexes, 2)).astype('float32') 
        # indices = np.arange(freqs.shape[1])
        # # for m in range(freqs.shape[0]):
        # #     np.random.shuffle(indices)
        # #     for n in range(grp):
        # #         # idx[n, m, :] = indices[n*numindexes:(n+1)*numindexes]
        # #         # freq[n] = freqs[:, indices[n*numindexes:(n+1)*numindexes], :]
        # #         print('freq shape', freq.shape)
        # # print('idx shape', idx.shape)
        
        psi_all_imag = np.zeros((grp,original.shape[1],original.shape[1]), dtype='float32')
        for t in range(grp):
            print(t)
            # Then what we see at the detector is the wave propagation
            # of the near field wavefront
            data = tike.ptycho.simulate(detector_shape=pw*2,
                                        probe=probe, scan=true_scan,
                                        psi=original,
                                        x=freq[t],)
            np.random.seed(1300)
            data = np.random.poisson(data)
            print(np.mean(data))
            print('data shape', data.shape)
            result = {
                'psi': np.zeros(original.shape, dtype='complex64') + np.exp(1j),
                'probe': probe,
                'scan': true_scan,
                'λ': 0,  # parameter for ADMM
                'μ': 0,  # parameter for ADMM
            }

            logging.basicConfig(level=logging.INFO)

            for i in range(1):
                result = tike.ptycho.reconstruct(
                    data= data.copy(),
                    x=freq[t].copy(),
                    **result,
                    algorithm='combined',
                    model = 'poisson',
                    num_iter=50,
                    recover_probe=False,
                    recover_positions=False,
                )
                # psi_all[t] = result['psi'].copy()
                # plt.figure()
                # tike.view.plot_phase(result['psi'][ntheta // 2], amin=0, amax=1)
                # plt.figure()
                # tike.view.plot_phase(original[0], amin=0)
                
                # plt.show()
                # for m in range(probe.shape[-3]):
                #     plt.figure()
                #     tike.view.plot_phase(result['probe'][0, 0, 0, m], amin=0)
                # plt.show()
                # dxchange.write_tiff(np.abs(result['psi'][ntheta // 2]), 'results/T'+name+'/psi_real')
                dxchange.write_tiff(np.angle(result['psi'][ntheta // 2]), 'results/T'+name+'/psi_imag')
                # plt.figure()
                # plt.subplot(1, 2, 1)
                # plt.imshow(np.abs(result['psi'][ntheta // 2]), vmin=0, vmax=1, cmap=plt.cm.twilight)
                # cb1 = plt.colorbar(orientation='horizontal')
                # plt.subplot(1, 2, 2)
                # plt.imshow(np.angle(result['psi'][ntheta // 2]), vmin=0, vmax=1, cmap=plt.cm.twilight)
                # cb0 = plt.colorbar(orientation='horizontal')
                # plt.imsave('results/T'+name+'/T'+name+'.png',np.angle(result['psi'][ntheta // 2]), vmin=0, vmax=1, cmap=plt.cm.twilight)

                psi_all_imag[t]  = psi_all_imag[t] + np.angle(result['psi'][ntheta // 2]).copy()

        print(np.max(psi_all_imag), psi_all_imag.shape)
        dxchange.write_tiff(np.mean(psi_all_imag, axis=0), 'results/T'+name+'/psi_imag_mean')
        dxchange.write_tiff(psi_all_imag, 'results/T'+name+'/psi_T'+name+'_imag')






#comment to run in denoiser         
# zpython ./main.py -dfn=dataset/v6/psi_T6Cone512paddedPoissongrp8noise1prbFscanFiter200_imag.tiff -ain=0 -expName="cone512Poiss-grp8noise1iter200ain0" &
# zpython ./main.py -dfn=dataset/v6/psi_T6Cone512paddedPoissongrp8noise1prbFscanFiter300_imag.tiff -ain=0 -expName="cone512Poiss-grp8noise1iter300ain0" &
# zpython ./main.py -dfn=dataset/v6/psi_T6Cone512paddedgrp8noise1prbFscanFiter400_imag.tiff -ain=0 -expName="cone512Gauss-grp8noise1iter400ain0" &