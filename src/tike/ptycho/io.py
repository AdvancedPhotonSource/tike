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

