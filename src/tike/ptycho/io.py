__author__ = "Tekin Bicer, Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

from tike.constants import wavelength


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
    photon_energy : float [keV]
        The energy of the incident beam.

    Returns
    -------
    positions : float [pixels]
        The scanning positions in pixel coordinates.

    """
    return positions * ((detector_pixel_width * detector_pixel_count) /
                        (detector_distance * wavelength(energy)))


def read_aps_2idd(diffraction_path, position_path):
    """Load ptychography data collected at the Advanced Photon Source 2-ID-D.

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
    data : (..., FRAME, WIDE, HIGH) float32
        Diffraction patterns; cropped square and centered on peak.
    scan : (..., POSI, 2) float32
        Scan positions; rescaled to pixel coordinates but uncentered.

    """
    import h5py

    scan = np.genfromtxt(position_path, delimiter=",")

    with h5py.File(parameter_path, 'r') as f:
        photon_energy = f['/entry/instrument/detector'
                          '/detectorSpecific/photon_energy'][()] / 1000.  # keV
        detect_width = f['/entry/instrument/detector'
                         '/detectorSpecific/x_pixels_in_detector'][()]
        detect_height = f['/entry/instrument/detector'
                          '/detectorSpecific/y_pixels_in_detector'][()]
        detector_dist = f['/entry/instrument/detector'
                          '/detector_distance'][()]  # meter
        det_pix_width = f['/entry/instrument/detector'
                          '/x_pixel_size'][()]  # meter
        beam_center_x = f['/entry/instrument/detector/beam_center_x'][()]
        beam_center_y = f['/entry/instrument/detector/beam_center_y'][()]

        radius = 256
        assert beam_center_x + radius < detect_width
        assert beam_center_y + radius < detect_height
        assert beam_center_x - radius >= 0
        assert beam_center_y - radius >= 0

        # TODO: Should be able to predict the number of datasets. However,
        # let's just catch exception for now.
        data = []

        def crop_diffraction(x):
            data.append(x[:, beam_center_x - radius:beam_center_x + radius,
                          beam_center_y - radius:beam_center_y + radius])

        with h5py.File('fly145_master.h5', 'r') as f:
            for x in f['/entry/data']:
                try:
                    crop_diffraction(f[f'/entry/data/{x}'])
                except KeyError:
                    break
            data = np.concatenate(data, axis=0)

    assert len(data) == len(scan), ("Number of positions and frames should be "
                                    f"equal not {data.shape}, {scan.shape}")

    scan = position_units_to_pixels(
        scan,
        detector_dist,
        data.shape[-1],
        det_pix_width,
        photon_energy,
    )

    return data.astype('float32'), scan.astype('float32')
