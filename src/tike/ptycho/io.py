__author__ = "Tekin Bicer, Daniel Ching"
__copyright__ = "Copyright (c) 2021, UChicago Argonne, LLC."

import warnings
import logging

import h5py
import numpy as np

from tike.constants import wavelength

logger = logging.getLogger(__name__)


def position_units_to_pixels(
    positions,
    detector_distance,
    detector_pixel_count,
    detector_pixel_width,
    photon_energy,
):
    """Convert scanning positions units from meters to pixels coordinates.

    Parameters
    ----------
    positions : float [m]
        Coordinates of the position of the beam when frames were collected.
    detector_distance : float [m]
        The propagation distance of the beam from the sample to the detector.
    detector_pixel_count : int
        The number of pixels across one edge of the detector. Assumes a square
        detector.
    detector_pixel_width : float [m]
        The width of one detector pixel. Assumes square pixels.
    photon_energy : float [eV]
        The energy of the incident beam.

    Returns
    -------
    positions : float [pixels]
        The scanning positions in pixel coordinates.

    """
    return positions * (
        (detector_pixel_width * detector_pixel_count) /
        (detector_distance * wavelength(photon_energy / 1000) / 100))


def read_aps_velociprobe(diffraction_path, position_path):
    """Load ptychography data from the Advanced Photon Source Velociprobe.

    Expects one HDF5 file and one CSV file with the following organization

    diffraction_path:
        /entry
            /data
                /data_000000:int[FRAME, WIDE, HIGH] {unit: counts}
                /data_000001:int[FRAME, WIDE, HIGH] {unit: counts}
                ...
            /instrument
                /detector
                    /beam_center_x:float            {unit: pixel}
                    /beam_center_y:float            {unit: pixel}
                    /detectorSpecific
                        /photon_energy:float        {unit: eV}
                        /x_pixels_in_detector:int
                        /y_pixels_in_detector:int
                    /detector_distance:float        {unit: m}
                    /x_pixel_size:float             {unit: m}

    Where FRAME is the number of detector frames recorded and WIDE/HIGH is the
    width and height. The number of data_000000 links may be more than the
    actual number of files because of some problem where the master file is
    created before the linked files are created.

    The CSV position raw data file is a 8 column file with columns
    corresponding to the following parameters: samz, samx, samy, zpx, zpy,
    encoder y, encoder x, trigger number. However, we don't use this file.
    Instead we use a preprocessed file with no header and two colums: the
    horiztonal and vertical positions.

    Parameters
    ----------
    diffraction_path : string
        The absolute path to the HDF5 file containing diffraction patterns and
        other metadata.
    position_path : string
        The absolute path to the CSV file containing position information.

    Returns
    -------
    data : (1, FRAME, WIDE, HIGH) float32
        Diffraction patterns; cropped square and peak FFT shifted to corner.
    scan : (1, POSI, 2) float32
        Scan positions; rescaled to pixel coordinates but uncentered.

    """
    with h5py.File(diffraction_path, 'r') as f:
        photon_energy = f['/entry/instrument/detector'
                          '/detectorSpecific/photon_energy'][()]  # eV
        detect_width = int(f['/entry/instrument/detector'
                             '/detectorSpecific/x_pixels_in_detector'][()])
        detect_height = int(f['/entry/instrument/detector'
                              '/detectorSpecific/y_pixels_in_detector'][()])
        detector_dist = f['/entry/instrument/detector'
                          '/detector_distance'][()]  # meter
        det_pix_width = f['/entry/instrument/detector'
                          '/x_pixel_size'][()]  # meter
        beam_center_x = int(f['/entry/instrument/detector/beam_center_x'][()])
        beam_center_y = int(f['/entry/instrument/detector/beam_center_y'][()])
        logger.info('Loading 2-ID-D ptychography data:\n'
                    f'\tphoton energy {photon_energy} eV\n'
                    f'\twidth: {detect_width}, center: {beam_center_x}\n'
                    f'\theight: {detect_height}, center: {beam_center_y}')

        # Autodetect the diffraction pattern size by doubling until it
        # doesn't fit on the detector anymore.
        radius = 2
        while (beam_center_x + radius < detect_width
               and beam_center_y + radius < detect_height
               and beam_center_x - radius >= 0 and beam_center_y - radius >= 0):
            radius *= 2
        radius = radius // 2
        logger.info(f'Autodetected diffraction size is {2* radius}.')

        data = []

        def crop_and_shift(x):
            data.append(
                np.fft.ifftshift(
                    x[..., beam_center_y - radius:beam_center_y + radius,
                      beam_center_x - radius:beam_center_x + radius],
                    axes=(-2, -1),
                ))

        for x in f['/entry/data']:
            try:
                crop_and_shift(f[f'/entry/data/{x}'])
            except KeyError:
                # Catches links to non-files.
                # TODO: Should be able to predict the number of datasets.
                # However, let's just catch exception for now.
                break
            except OSError as error:
                warnings.warn(
                    "The HDF5 compression plugin is probably missing. "
                    "See the conda-forge hdf5-external-filter-plugins package.")
                raise error

        data = np.concatenate(data, axis=0)

    scan = np.genfromtxt(position_path, delimiter=",")
    logging.info(f'Loaded {len(scan)} scan positions.')

    if len(data) != len(scan):
        warnings.warn(
            f"The number of positions {data.shape} and frames {scan.shape}"
            " is not equal. One of the two will be truncated.")
        num_frame = min(len(data), len(scan))
        scan = scan[:num_frame, ...]
        data = data[:num_frame, ...]

    scan = position_units_to_pixels(
        scan,
        detector_dist,
        data.shape[-1],
        det_pix_width,
        photon_energy,
    )

    data = data[None, ...]
    scan = scan[None, ...]

    return data.astype('float32'), scan.astype('float32')
