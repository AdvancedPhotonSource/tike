"""A prototype implementation of a multi-slice propagation operation."""

import numpy as np


def forward(
    object: np.ndarray[np.complex64],
    probe: np.ndarray[np.complex64],
    spacing: float
) -> np.ndarray[np.complex64]:
    """The forward multi-slice propagation operator.

    Parameters
    ----------
    object: (S, H, W)
        A sequence of `S` object slices
    probe: (H, W)
        A illumination/wavefront before propagation through the `object slices`
    spacing: float [pixels]
        The distance between object slices in pixel units

    Returns
    -------
    wavefront: (H, W)
        A illumination/wavefront after propagation through the `object slices`

    """
    assert object.ndim == 3, "The object should have three dimensions"
    assert probe.ndim == 2, "The probe should have two dimensions"
    assert object.shape[-2:] == probe.shape[
        -2:], "For this example, the object and probe should have the same shape."

    wavefront = probe

    return wavefront
