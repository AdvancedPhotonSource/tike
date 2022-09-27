import os

import h5py
import tike.operators
import tike.ptycho
import tike.view
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

_dir = os.path.dirname(__file__)


def test_wavefronts():
    """Test that forward model (diffraction and propagation) is consistent.

    Data taken from reconstruction with 2 probes and 5 eigen modes.
    """
    with h5py.File(os.path.join(_dir, 'forward-in1.mat'), 'r') as inputs:

        probes = np.concatenate(
            (
                inputs['probe0'],
                inputs['probe1'],
            ),
            axis=1,
        )[None, ...].view('complex64')
        print(probes.shape, probes.dtype)
        assert probes.shape == (1, 6, 2, 512, 512)
        shared_probes = probes[:, :1, :, :, :]
        eigen_probes = probes[:, 1:, :1, :, :]
        print(shared_probes.shape)
        print(eigen_probes.shape)
        weights = inputs['probe_evolution'][...].T[..., None].astype('float32')
        print(weights.shape, weights.dtype)
        assert weights.shape == (63, 6, 1)
        psi = inputs['object'][...].view('complex64')
        print(psi.shape, psi.dtype)
        positions0 = inputs['positions0'][0].astype('float32')
        print(positions0.shape, positions0.dtype)
        assert positions0.shape == (63,)
        positions1 = inputs['positions1'][0].astype('float32')
        print(positions1.shape, positions1.dtype)
        assert positions1.shape == (63,)

    with h5py.File(os.path.join(_dir, 'forward-out.mat'), 'r') as outputs:
        patches = outputs['obj_proj_'][...].view('complex64')
        print(patches.shape, patches.dtype)
        probe0 = outputs['probe0'][...][:, None, None].view('complex64')
        probe1 = outputs['probe1'][...][None, None, None, ...].view('complex64')
        print(probe0.shape, probe0.dtype)
        print(probe1.shape, probe1.dtype)
    with h5py.File(os.path.join(_dir, 'forward-out1.mat'), 'r') as outputs:
        psi1 = outputs['psi1'][...][:, None, None, ...].view('complex64')
        print(psi1.shape, psi1.dtype)
        assert psi1.shape == (63, 1, 1, 512, 512)
    with h5py.File(os.path.join(_dir, 'forward-out2.mat'), 'r') as outputs:
        psi2 = outputs['psi2'][...][:, None, None, ...].view('complex64')
        print(psi2.shape, psi2.dtype)
        assert psi2.shape == (63, 1, 1, 512, 512)

    # Test that probe combination is consistent

    varying_probe = tike.ptycho.probe.get_varying_probe(
        shared_probes,
        eigen_probes,
        weights=weights,
    )
    print(varying_probe.shape, varying_probe.dtype)

    plt.figure()
    tike.view.plot_complex(probe0[5, 0, 0])
    plt.savefig('forward-02.png')

    plt.figure()
    tike.view.plot_complex((varying_probe[5, 0, 0]))
    plt.savefig('forward-03.png')

    plt.figure()
    tike.view.plot_complex(probe1[0, 0, 0])
    plt.savefig('forward-04.png')

    plt.figure()
    tike.view.plot_complex((varying_probe[0, 0, 1]))
    plt.savefig('forward-05.png')

    np.testing.assert_allclose(
        probe0,
        varying_probe[:, :, :1],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        probe1,
        varying_probe[:1, :1, 1:2],
        atol=1e-6,
    )

    # Test that patch extraction and wave propagation is consistent

    positions = np.stack((positions1, positions0), axis=1)
    assert positions.shape == (63, 2)

    with tike.operators.Ptycho(
            detector_shape=shared_probes.shape[-1],
            probe_shape=shared_probes.shape[-1],
            model='gaussian',
            nz=1,
            n=1,
    ) as op:

        new_patches = op.diffraction.patch.fwd(
            cp.asarray(psi, dtype='complex64', order='C'),
            cp.asarray(positions, dtype='float32', order='C'),
            patch_width=512,
        ).get()

        print(new_patches.shape, new_patches.dtype)

        wavefronts = op.diffraction.fwd(
            probe=cp.asarray(varying_probe[..., 0, :, :, :],
                             dtype='complex64',
                             order='C'),
            scan=cp.asarray(positions, dtype='float32', order='C'),
            psi=cp.asarray(psi, dtype='complex64', order='C'),
        ).get()

        # farfield = op.fwd(
        #     probe=cp.asarray(varying_probe, dtype='complex64', order='C'),
        #     scan=cp.asarray(positions, dtype='float32', order='C'),
        #     psi=cp.asarray(psi, dtype='complex64', order='C'),
        # ).get()

    plt.figure()
    plt.imshow(cp.asarray(psi, dtype='complex64').real.get())
    plt.savefig('forward-00.png')

    plt.figure()
    for i in range(5):
        plt.subplot(5, 3, 3 * i + 1)
        plt.imshow(patches[i].real)
        plt.subplot(5, 3, 3 * i + 2)
        plt.imshow(new_patches[i].real)
        plt.subplot(5, 3, 3 * i + 3)
        plt.imshow(new_patches[i].real - patches[i].real)
        plt.colorbar()
    plt.savefig('forward-01.png')

    np.testing.assert_allclose(
        new_patches,
        patches,
        atol=1e-6,
    )

    plt.figure()
    tike.view.plot_complex(wavefronts[0,0])
    plt.savefig('forward-08.png')
    plt.figure()
    tike.view.plot_complex(psi1[0,0,0])
    plt.savefig('forward-09.png')
    plt.figure()
    tike.view.plot_complex(wavefronts[0,0] - psi1[0,0,0])
    plt.savefig('forward-10.png')

    # NOTE: MATLAB uses naive complex multiplication, but standard CUDA library
    # uses optimized (but still correct) complex multiplication. This accounts
    # for diverence in behavior.

    np.testing.assert_allclose(
        wavefronts[:, None, :1],
        psi1,
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        wavefronts[:, None, 1:2],
        psi2,
        rtol=1e-6,
    )

    # NOTE: MATLAB and CUPY FFTs seem qualitatively the same, but maximum
    # relative error is 2%

    # plt.figure()
    # tike.view.plot_complex(np.fft.ifftshift(farfield[11, 0, 0]))
    # plt.savefig('forward-11.svg') plt.figure()
    # tike.view.plot_complex(np.fft.ifftshift(psi1[11, 0, 0]))
    # plt.savefig('forward-12.svg') plt.figure()
    # tike.view.plot_complex(np.fft.ifftshift(farfield[11, 0, 0] - psi1[11, 0,
    # 0])) plt.savefig('forward-13.svg')

    # np.testing.assert_allclose(
    #     farfield[:, :, :1],
    #     psi1,
    #     rtol=1e-2,
    # )

    # np.testing.assert_allclose(
    #     farfield[:, :, 1:2],
    #     psi2,
    #     rtol=1e-2,
    # )
