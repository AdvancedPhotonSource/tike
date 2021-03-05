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


def read_aps_2idd(diffraction_path, parameter_path):
    """Load ptychogrpahy data collected at the Advanced Photon Source 2-ID-D.

    Expects two HDF5 files with the following organization

    diffraction_path.h5:
        /df:int[FRAME, WIDE, HIGH] {unit: counts}

    parameter_path.h5:
        /entry
            /instrument
                /detector
                    /detectorSpecific
                        /photon_energy:float {unit: eV}
                    /detector_distance:float {unit: mm}
                    /x_pixel_size:float {unit: m}
                    /beam_center_x:float[POSI] {unit: ?}
                    /beam_center_y:float[POSI] {unit: ?}

    Where FRAME is the number of detector frames recorded, POSI is the
    number of scan positions recorded, WIDE/HIGH is the width and height.

    Parameters
    ----------
    diffraction_path : string
        The absolute path to the HDF5 file containing diffraction patterns.
    parameter_path : string
        The absolute path to the HDF5 file containing position information.

    Returns
    -------
    data : (..., FRAME, WIDE, HIGH) float32
        Diffraction patterns; cropped square and centered on peak.
    scan : (..., POSI, 2) float32
        Scan positions; rescaled to pixel coordinates but uncentered.
    """
    import h5py

    with h5py.File(parameter_path, 'r') as f:
        photon_energy = f['/entry/instrument/detector'
                          '/detectorSpecific/photon_energy'][()] / 1000.  # keV
        detector_dist = f['/entry/instrument/detector'
                          '/detector_distance'][()] / 1000.  # meter
        det_pix_width = f['/entry/instrument/detector'
                          '/x_pixel_size'][()]  # meter
        scan_coords_x = f['/entry/instrument/detector/beam_center_x'][()]
        scan_coords_y = f['/entry/instrument/detector/beam_center_y'][()]

    with h5py.File(diffraction_path, 'r') as f:
        data = f['/dp'][()]
        # TODO: FFT shift and crop frames to square shape?

    scan = position_units_to_pixels(
        np.stack([scan_coords_x, scan_coords_y], axis=1),
        detector_dist,
        data.shape[-1],
        det_pix_width,
        photon_energy,
    )

    return data.astype('float32'), scan.astype('float32')
