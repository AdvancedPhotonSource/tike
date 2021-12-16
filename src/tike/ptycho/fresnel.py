"""Functions for probe initialization using a model of Fresnel optics."""
import numpy as np


def single_probe(
    probe_shape,
    lambda0,
    dx,
    dis_defocus,
    zone_plate_params,
):
    """Estimate the probe using Fresnel propagation model of focusing optics.

    Example
    -------

    .. code-block:: python

        single_probe(
            probe_shape=64
            lambda0=1.24e-9 / 10
            dx=6e-6,
            dis_defocus=800e-6
            zone_plate_params=dict(
                radius=150e-6 / 2,
                outmost=50e-9,
                beamstop=60e-6,
            ),
        )

    Parameters
    ----------
    probe_shape : int
        The pixel width and height of the (square) probe illumination.
    lambda0 : float [m]
        The central wavelength of the illumination.
    dx : float [m]
        The pixel size on sample plane
    dis_defocus : float [m]
        The defocus distance; the distance from the sample to the focal plane.
        May be negative.
    dis_StoD : float [m]
        The sample to detector distance.
    zone_plate_params: str or dict
        One of 'velo', '2idd', 'lamni' or a dictionary with the following keys:
        radius: float [m] zone plate radius, outmost : float [m] outmost zone
        width, and beamstop : float [m] diameter of central beamstop.

    Returns
    -------
    probe : (1, 1, SHARED, WIDE, HIGH) complex64
        An estimate of the probe.
    """

    # get zone plate parameter
    T, dx_fzp, FL0 = _fzp_calculate(lambda0, dis_defocus, probe_shape, dx,
                                    zone_plate_params)

    probe = _fresnel_propagation(T, dx_fzp, (FL0 + dis_defocus), lambda0)

    probe = probe / (np.sqrt(np.sum(np.abs(probe)**2)))

    return probe[np.newaxis, np.newaxis, np.newaxis].astype(np.complex64)


def MW_probe(
    probe_shape,
    lambda0,
    dx,
    dis_defocus,
    zone_plate_params,
    energy=1,
    bandwidth=0.01,
    spectrum=None,
):
    """Estimate multi-energy probes using Fresnel propagation model of optics.

    Example
    -------

    .. code-block:: python

        MW_probe(
            probe_shape=64
            lambda0=1.24e-9 / 10
            dx=6e-6
            dis_defocus=800e-6
            dis_StoD=2
            zone_plate_params=dict(
                radius=150e-6 / 2,
                outmost=50e-9,
                beamstop=60e-6,
            ),
            energy=5,
            bandwidth=0.01,
        )

    Parameters
    ----------
    probe_shape : int
        The pixel width and height of the (square) probe illumination.
    lambda0 : float [m]
        The central wavelength of the illumination.
    dx : float [m]
        The pixel size on sample plane
    dis_defocus : float [m]
        The defocus distance; the distance from the sample to the focal plane.
        May be negative.
    zone_plate_params: str or dict
        One of 'velo', '2idd', 'lamni' or a dictionary with the following keys:
        radius: float [m] zone plate radius, outmost : float [m] outmost zone
        width, and beamstop : float [m] diameter of central beamstop.
    energy : int
        number of energies for multi-wavelength method
    spectrum : [(wavelength, intensity), (wavelength, intensity), ...]
        A 2-tuple of wavelength and intensity for each energy. Assumes spectrum
        provided in ascending order by wavelength.
    bandwidth : float [m]
        The full width at half maximum of the spectrum divided by the central
        wavelength.

    Returns
    -------
    probe : (1, 1, SHARED, WIDE, HIGH) complex64
        An estimate of the probes sorted by spectrum.
    """
    if spectrum is None:
        spectrum = _gaussian_spectrum(lambda0, bandwidth, energy)
    else:
        # Trim the spectrum down the desired number of energies
        # FIXME: Sort by wavelength first; assumes wavelengths in order
        spectrum = spectrum[::spectrum.shape[0] // energy, :][:energy, :]
        # The central wavelength becomes the peak spectrum
        lambda0 = spectrum[np.argmax(spectrum[1, :]), 0]

    spectrum = spectrum[np.argsort(-spectrum[:, 1])]

    # focal length for central wavelength
    _, _, FL0 = _fzp_calculate(spectrum[0, 0], dis_defocus, probe_shape, dx,
                               zone_plate_params)

    probe = []
    for i in range(energy):
        # get zone plate parameter
        T, dx_fzp, _ = _fzp_calculate(spectrum[i, 0], dis_defocus, probe_shape,
                                      dx, zone_plate_params)

        nprobe = _fresnel_propagation(T, dx_fzp, (FL0 + dis_defocus),
                                      spectrum[i, 0])

        nprobe = nprobe / (np.sqrt(np.sum(np.abs(nprobe)**2)))

        probe.append(nprobe * (np.sqrt(spectrum[i, 1])))

    probe = np.stack(probe, axis=0)
    return probe[np.newaxis, np.newaxis].astype('complex64')


def _gaussian_spectrum(lambda0, bandwidth, energy):
    spectrum = np.zeros((energy, 2))
    sigma = lambda0 * bandwidth / 2.355
    d_lam = sigma * 4 / (energy - 1)
    spectrum[:, 0] = np.arange(-1 * np.floor(energy / 2), np.ceil(
        energy / 2)) * d_lam + lambda0
    spectrum[:, 1] = np.exp(-(spectrum[:, 0] - lambda0)**2 / sigma**2)
    return spectrum


def _fzp_calculate(wavelength, dis_defocus, M, dx, zone_plate_params):
    """
    this function can calculate the transfer function of zone plate
    return the transfer function, and the pixel sizes
    """

    FZP_para = _get_setup(zone_plate_params)

    FL = 2 * FZP_para['radius'] * FZP_para['outmost'] / wavelength

    # pixel size on FZP plane
    dx_fzp = wavelength * (FL + dis_defocus) / M / dx
    # coordinate on FZP plane
    lx_fzp = -dx_fzp * np.arange(-1 * np.floor(M / 2), np.ceil(M / 2))

    XX_FZP, YY_FZP = np.meshgrid(lx_fzp, lx_fzp)
    # transmission function of FZP
    T = np.exp(-1j * 2 * np.pi / wavelength * (XX_FZP**2 + YY_FZP**2) / 2 / FL)
    C = np.sqrt(XX_FZP**2 + YY_FZP**2) <= FZP_para['radius']
    H = np.sqrt(XX_FZP**2 + YY_FZP**2) >= FZP_para['beamstop'] / 2

    return T * C * H, dx_fzp, FL


def _get_setup(zone_plate_params):

    if isinstance(zone_plate_params, str):
        switcher = {
            'velo': {
                'radius': 90e-6,
                'outmost': 50e-9,
                'beamstop': 60e-6
            },
            '2idd': {
                'radius': 80e-6,
                'outmost': 70e-9,
                'beamstop': 60e-6
            },
            'lamni': {
                'radius': 114.8e-6 / 2,
                'outmost': 60e-9,
                'beamstop': 40e-6
            },
        }
        if zone_plate_params in switcher:
            return switcher[zone_plate_params]
        else:
            raise ValueError(
                f"{zone_plate_params} is not a known zone plate. "
                f"Choose one of {switcher.keys()} or provide a dictionary "
                "with custom zone plate parameters.")

    return zone_plate_params


def _fresnel_propagation(input, dxy, z, wavelength):
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
    dx = lambda0 * dis_StoD / shape / dx_dec

    # test single probe modes
    probe = single_probe(shape,
                         lambda0,
                         dx,
                         dis_defocus,
                         zone_plate_params=dict(
                             radius=150e-6 / 2,
                             outmost=50e-9,
                             beamstop=60e-6,
                         ))

    print(probe.shape)
    plt.figure(1)
    pb = probe[..., :, :].squeeze()
    plt.imshow(np.abs(pb))
    plt.show()

    # test multi-wavelength probe modes
    MW_probe = MW_probe(shape,
                        lambda0,
                        dx,
                        dis_defocus,
                        zone_plate_params='2idd',
                        energy=5,
                        bandwidth=0.01)

    print(MW_probe.shape)
    for i in range(5):
        plt.figure()
        pb = MW_probe[..., i, :, :].squeeze()
        plt.imshow(np.abs(pb))
    plt.show()
