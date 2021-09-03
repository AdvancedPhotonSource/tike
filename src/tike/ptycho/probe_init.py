# This is the python version code for initialize probes

import numpy as np


def single_probe(probe_shape, lambda0, dx_dec, dis_defocus, dis_StoD, **kwargs):
    # return probe sorted by the spectrum
    # return scale is the wavelength dependent pixel scaling factor
    """
    Summary of this function goes here
    Parameters: probe_shape  -> the matrix size for probe
                lambda0      -> central wavelength
                dx_dec       -> pixel size on detector
                dis_defocus  -> defocus distance (sample to the focal plane)
                dis_StoD     -> sample to detector distance
                kwargs       -> setup: 'velo','2idd','lamni'
                             -> radius: zone plate radius
                             -> outmost: outmost zone width
                             -> beamstop: diameter of central beamstop
    """

    probe = np.zeros((probe_shape, probe_shape), dtype=np.complex)

    # pixel size on sample plane
    dx = lambda0 * dis_StoD / probe_shape / dx_dec

    # get zone plate parameter
    T, dx_fzp, FL0 = fzp_calculate(lambda0, dis_defocus, probe_shape, dx,
                                   **kwargs)

    nprobe = fresnel_propagation(T, dx_fzp, (FL0 + dis_defocus), lambda0)

    probe = nprobe / (np.sqrt(np.sum(np.abs(nprobe)**2)))

    return probe[np.newaxis, np.newaxis, np.newaxis, np.newaxis]


def MW_probe(probe_shape, energy, lambda0, dx_dec, dis_defocus, dis_StoD,
             **kwargs):
    # return probe sorted by the spectrum
    # return scale is the wavelength dependent pixel scaling factor
    """
    Summary of this function goes here
    Parameters: probe_shape  -> the matrix size for probe
                energy       -> number of energies for multi-wavelength method
                lambda0      -> central wavelength
                dx_dec       -> pixel size on detector
                dis_defocus  -> defocus distance (sample to the focal plane)
                dis_StoD     -> sample to detector distance
                kwargs       -> setup: 'velo','2idd','lamni'
                             -> radius: zone plate radius
                             -> outmost: outmost zone width
                             -> beamstop: diameter of central beamstop
                             -> spectrum: measured spectrum (if available)
                                [wavelength,intensity]
                             -> bandwidth
    """

    if 'spectrum' in kwargs:
        spectrum = kwargs.get('spectrum')
        spectrum = spectrum[::spectrum.shape[0] // energy, :][:energy, :]
        lambda0 = spectrum[np.argmax(spectrum[1, :]), 0]
    else:
        if 'bandwidth' in kwargs:
            bandwidth = kwargs.get('bandwidth')
        else:
            bandwidth = 0.01

        spectrum = gaussian_spectrum(lambda0, bandwidth, energy)

    spectrum = spectrum[np.argsort(-spectrum[:, 1])]

    probe = np.zeros((energy, 1, probe_shape, probe_shape), dtype=np.complex)

    # pixel size on sample plane (central wavelength)
    dx = spectrum[0, 0] * dis_StoD / probe_shape / dx_dec

    # focal length for central wavelength
    _, _, FL0 = fzp_calculate(spectrum[0, 0], dis_defocus, probe_shape, dx,
                              **kwargs)

    for i in range(energy):
        # get zone plate parameter
        T, dx_fzp, _ = fzp_calculate(spectrum[i, 0], dis_defocus, probe_shape,
                                     dx, **kwargs)

        nprobe = fresnel_propagation(T, dx_fzp, (FL0 + dis_defocus),
                                     spectrum[i, 0])

        nprobe = nprobe / (np.sqrt(np.sum(np.abs(nprobe)**2)))

        probe[i, 0, :, :] = nprobe * (np.sqrt(spectrum[i, 1]))

    return probe[np.newaxis, np.newaxis]


def gaussian_spectrum(lambda0, bandwidth, energy):
    spectrum = np.zeros((energy, 2))
    sigma = lambda0 * bandwidth / 2.355
    d_lam = sigma * 4 / (energy - 1)
    spectrum[:, 0] = np.arange(-1 * np.floor(energy / 2), np.ceil(
        energy / 2)) * d_lam + lambda0
    spectrum[:, 1] = np.exp(-(spectrum[:, 0] - lambda0)**2 / sigma**2)
    return spectrum


def fzp_calculate(wavelength, dis_defocus, M, dx, **kwargs):
    """
    this function can calculate the transfer function of zone plate
    return the transfer function, and the pixel sizes
    """

    FZP_para = get_setup(**kwargs)

    FL = 2 * FZP_para['radius'] * FZP_para['outmost'] / wavelength

    # pixel size on FZP plane
    dx_fzp = wavelength * (FL + dis_defocus) / M / dx
    # coordinate on FZP plane
    lx_fzp = -dx_fzp * np.arange(-1 * np.floor(M / 2), np.ceil(M / 2))

    XX_FZP, YY_FZP = np.meshgrid(lx_fzp, lx_fzp)
    # transmission function of FZP
    T = np.exp(-1j * 2 * np.pi / wavelength * (XX_FZP**2 + YY_FZP**2) / 2 / FL)
    C = np.sqrt(XX_FZP**2 + YY_FZP**2) <= FZP_para['radius']
    H = np.sqrt(XX_FZP**2 + YY_FZP**2) >= FZP_para['CS'] / 2

    return T * C * H, dx_fzp, FL


def get_setup(**kwargs):

    if 'setup' in kwargs:
        setup = kwargs.get('setup')
    else:
        setup = 'custom'

    switcher = {
        'velo': {
            'radius': 90e-6,
            'outmost': 50e-9,
            'CS': 60e-6
        },
        '2idd': {
            'radius': 80e-6,
            'outmost': 70e-9,
            'CS': 60e-6
        },
        'lamni': {
            'radius': 114.8e-6 / 2,
            'outmost': 60e-9,
            'CS': 40e-6
        },
        'custom': {
            'radius': kwargs.get('radius'),
            'outmost': kwargs.get('outmost'),
            'CS': kwargs.get('beamstop')
        }
    }

    FZP_para = switcher.get(setup)
    return FZP_para


def fresnel_propagation(input, dxy, z, wavelength):
    """
    This is the python version code for fresnel propagation
    Summary of this function goes here
    Parameters:    dx,dy  -> the pixel pitch of the object
                z      -> the distance of the propagation
                lambda -> the wave length
                X,Y    -> meshgrid of coordinate
                input     -> input object
    """

    (M, N) = input.shape
    k = 2 * np.pi / wavelength
    # the coordinate grid
    M_grid = np.arange(-1 * np.floor(M / 2), np.ceil(M / 2))
    N_grid = np.arange(-1 * np.floor(N / 2), np.ceil(N / 2))
    lx = M_grid * dxy
    ly = N_grid * dxy

    XX, YY = np.meshgrid(lx, ly)

    # the coordinate grid on the output plane
    fc = 1 / dxy
    fu = wavelength * z * fc
    lu = M_grid * fu / M
    lv = N_grid * fu / N
    Fx, Fy = np.meshgrid(lu, lv)

    if z > 0:
        pf = np.exp(1j * k * z) * np.exp(1j * k * (Fx**2 + Fy**2) / 2 / z)
        kern = input * np.exp(1j * k * (XX**2 + YY**2) / 2 / z)
        cgh = np.fft.fft2(np.fft.fftshift(kern))
        OUT = np.fft.fftshift(cgh * np.fft.fftshift(pf))
    else:
        pf = np.exp(1j * k * z) * np.exp(1j * k * (XX**2 + YY**2) / 2 / z)
        cgh = np.fft.ifft2(
            np.fft.fftshift(input * np.exp(1j * k * (Fx**2 + Fy**2) / 2 / z)))
        OUT = np.fft.fftshift(cgh) * pf
    return OUT


if __name__ == "__main__":

    import matplotlib.pylab as plt

    shape = 64
    lambda0 = 1.24e-9 / 10
    dx_dec = 75e-6
    dis_defocus = 800e-6
    dis_StoD = 2

    # test single probe modes
    probe = single_probe(shape,
                         lambda0,
                         dx_dec,
                         dis_defocus,
                         dis_StoD,
                         radius=150e-6 / 2,
                         outmost=50e-9,
                         beamstop=60e-6)

    print(probe.shape)
    plt.figure(1)
    pb = probe[..., :, :].squeeze()
    plt.imshow(np.abs(pb))
    plt.show()

    # test multi-wavelength probe modes
    MW_probe = MW_probe(shape,
                        5,
                        lambda0,
                        dx_dec,
                        dis_defocus,
                        dis_StoD,
                        setup='2idd',
                        bandwidth=0.01)

    print(MW_probe.shape)
    plt.figure(2)
    for i in range(5):
        pb = MW_probe[..., i, 0, :, :].squeeze()
        plt.imshow(np.abs(pb))
        plt.show()
