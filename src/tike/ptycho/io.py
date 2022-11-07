"""Functions for loading and preprocessing raw data from the disk."""

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
    positions : array[float] [m]
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
    pixel_per_meter = (
        (detector_pixel_width * detector_pixel_count) /
        (detector_distance * wavelength(photon_energy / 1000) / 100))
    logger.info(
        "Based on detector distance and photon energy,"
        f" reconstruction pixel size will be {1 / pixel_per_meter:.3e} m.")
    return positions * pixel_per_meter


def read_aps_velociprobe(
        diffraction_path,
        position_path,
        xy_columns=(5, 1),
        trigger_column=7,
        max_crop=2048,
):
    """Load ptychography data from the Advanced Photon Source Velociprobe.

    Expects one HDF5 file and one CSV file with the following organization::

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
                /sample
                    /goniometer
                        /chi:float[]                    {unit: degree }

    Where FRAME is the number of detector frames recorded and WIDE/HIGH is the
    width and height. The number of data_000000 links may be more than the
    actual number of files because of some problem where the master file is
    created before the linked files are created.

    We use lz4 to compress the data. In order to open these compressed
    datasets, you have to install the lz4 filter plugins
    (https://github.com/nexusformat/HDF5-External-Filter-Plugins).

    The CSV position raw data file is a 8 column file with columns
    corresponding to the following parameters: samz, samx, samy, zpx, zpy,
    encoder y, encoder x, trigger number. Because the horizontal stage is on
    top of the rotation, stage, we must use the rotation stage position to
    correct the horizontal scanning positions. By default we use samx, encoder
    y for the horizontal and vertical positions.

    Parameters
    ----------
    diffraction_path : string
        The absolute path to the HDF5 file containing diffraction patterns and
        other metadata.
    position_path : string OR List[string]
        The absolute path to the CSV file(s) containing position information.
    xy_columns : 2-tuple of int
        The columns in the 8 column raw position file to use for x,y positions
    trigger_column : int
        The column in the 8 column raw position file to use for grouping
        positions together.
    max_crop : int
        Crop the diffraction pattern to at most this width.

    Returns
    -------
    data : (FRAME, WIDE, HIGH)
        Diffraction patterns; cropped square and peak FFT shifted to corner.
    scan : (POSI, 2) float32
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
        chi = float(f['entry/sample/goniometer/chi'][0])
        logger.info('Loading 2-ID-D ptychography data:\n'
                    f'\tstage rotation {chi} degrees\n'
                    f'\tphoton energy {photon_energy} eV\n'
                    f'\twidth: {detect_width}, center: {beam_center_x}\n'
                    f'\theight: {detect_height}, center: {beam_center_y}')

        # Autodetect the diffraction pattern size by doubling until it
        # doesn't fit on the detector anymore.
        max_radius = max_crop // 2
        radius = 2
        while (
            radius <= max_radius
            and beam_center_x + radius < detect_width
            and beam_center_y + radius < detect_height
            and beam_center_x - radius >= 0 and beam_center_y - radius >= 0
        ):  # yapf:disable
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

    # Load data from six column file
    if isinstance(position_path, list):
        raw_position = [np.genfromtxt(
            p,
            usecols=(*xy_columns, trigger_column),
            delimiter=',',
            dtype='int',
        ) for p in position_path]
        raw_position = np.concatenate(raw_position, axis=0)
    else:
        raw_position = np.genfromtxt(
            position_path,
            usecols=(*xy_columns, trigger_column),
            delimiter=',',
            dtype='int',
        )


    # Split positions where trigger number increases by 1. Assumes that
    # positions are ordered by trigger number in file. Shift indices by 1
    # because of how np.diff is defined.
    sections = np.nonzero(np.diff(raw_position[:, -1]))[0] + 1
    groups = np.split(
        raw_position[:, :-1],
        indices_or_sections=sections,
        axis=0,
    )

    # Apply a reduction function to handle multiple positions per trigger
    def position_reduce(g):
        """Average of the first and last position in each trigger group."""
        # return np.mean(g, axis=0, keepdims=True)
        return (g[:1] + g[-1:]) / 2

    groups = list(map(position_reduce, groups))
    scan = np.concatenate(groups, axis=0)

    # Rescale according to geometry of velociprobe
    scan[:, 0] *= -1e-9
    scan -= np.mean(scan, axis=0, keepdims=True)
    scan[:, 1] *= 1e-9 * np.cos(chi / 180 * np.pi)

    logger.info(f'Loaded {len(scan)} scan positions.')

    if len(data) != len(scan):
        warnings.warn(
            f"The number of positions {scan.shape} and frames {data.shape}"
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

    if np.any(np.logical_not(np.isfinite(data))):
        warnings.warn("Some values in the diffraction data are not finite. "
                      "Photon counts must be >= 0 and finite.")

    if np.any(data < 0):
        warnings.warn("Some values in the diffraction data are negative. "
                      "Photon counts must be >= 0 and finite.")

    return data, scan.astype('float32')

def read_aps_lynx(
    diffraction_path,
    position_path,
    photon_energy,
    beam_center_x,
    beam_center_y,
    detector_dist,
    xy_columns=(6, 3),
    trigger_column: int = 0,
    max_crop: int = 2048,
    gap_value: int = 2**12 - 1,
):
    """Load ptychography data from Advanced Photon Source LYNX.

    Expects one h5 file and one dat file with the following organization::

        diffraction_path:
            entry:NXentry
            @NX_class = NXentry
                data:NXdata
                @NX_class = NXdata
                eiger_4:NX_UINT16[12320,1030,1614] =
                    @Count_cutoff = 199998
                    @Exposure_period = 0.052000000000000005
                    @Exposure_time = 0.005
                    @Pixel_size = [7.5e-05]

    DAT is a space-separated text file with two rows of headers with positions
    in nanometers.

    Parameters
    ----------
    diffraction_path : string
        The absolute path to the HDF5 file containing diffraction patterns.
    position_path : string
        The absolute path to the CSV file containing position information.
    xy_columns : 2-tuple of int
        The columns in the 8 column raw position file to use for x,y positions
    trigger_column : int
        The column in the 8 column raw position file to use for grouping
        positions together.
    max_crop : int
        Crop the diffraction pattern to at most this width.
    gap_value : int
        The value used to encode detector gaps

    Returns
    -------
    data : (FRAME, WIDE, HIGH)
        Diffraction patterns; cropped square and peak FFT shifted to corner.
    scan : (POSI, 2) float32
        Scan positions; rescaled to pixel coordinates but uncentered.


    """
    with h5py.File(diffraction_path, 'r') as f:
        det_pix_width = f['/entry/data/eiger_4'].attrs['Pixel_size'].item()  # meter
        _, detect_height, detect_width = f['/entry/data/eiger_4'].shape
        logger.info('Loading 28-ID-C ptychography data:\n'
                    f'\tphoton energy {photon_energy} eV\n'
                    f'\twidth: {detect_width}, center: {beam_center_x}\n'
                    f'\theight: {detect_height}, center: {beam_center_y}\n'
                    f'\tdetector pixel width: {det_pix_width} m\n')


        # Autodetect the diffraction pattern size by doubling until it
        # doesn't fit on the detector anymore.
        max_radius = max_crop // 2
        radius = 2
        while (
            radius <= max_radius
            and beam_center_x + radius < detect_width
            and beam_center_y + radius < detect_height
            and beam_center_x - radius >= 0 and beam_center_y - radius >= 0
        ):  # yapf:disable
            radius *= 2
        radius = radius // 2
        logger.info(f'Autodetected diffraction size is {2 * radius}.')

        data = []

        def crop_and_shift(x):
            data.append(
                np.fft.ifftshift(
                    x[..., beam_center_y - radius:beam_center_y + radius,
                      beam_center_x - radius:beam_center_x + radius],
                    axes=(-2, -1),
                ))

        try:
            crop_and_shift(f['/entry/data/eiger_4'])
        except OSError as error:
            warnings.warn(
                "The HDF5 compression plugin is probably missing. "
                "See the conda-forge hdf5-external-filter-plugins package.")
            raise error

    data = np.concatenate(data, axis=0)
    # Set between panel values to zero
    data[data == gap_value] = 0

    raw_position = np.genfromtxt(
        position_path,
        usecols=(*xy_columns, trigger_column),
        delimiter=' ',
        dtype=np.float32,
        skip_header=2,
    )
    scan = raw_position[:, :2] * -1e-6

    logger.info(f'Loaded {len(scan)} scan positions.')

    if len(data) != len(scan):
        warnings.warn(
            f"The number of positions {scan.shape} and frames {data.shape}"
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

    if np.any(np.logical_not(np.isfinite(data))):
        warnings.warn("Some values in the diffraction data are not finite. "
                      "Photon counts must be >= 0 and finite.")

    if np.any(data < 0):
        warnings.warn("Some values in the diffraction data are negative. "
                      "Photon counts must be >= 0 and finite.")

    return data, scan.astype('float32')
