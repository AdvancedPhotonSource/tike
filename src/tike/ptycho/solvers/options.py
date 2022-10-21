import abc
import dataclasses
import typing

import numpy as np
import scipy.ndimage

from tike.ptycho.object import ObjectOptions
from tike.ptycho.position import PositionOptions, check_allowed_positions
from tike.ptycho.probe import ProbeOptions


@dataclasses.dataclass
class IterativeOptions(abc.ABC):
    """A base class providing options for iterative algorithms.

    .. versionadded:: 0.20.0
    """
    name: str = dataclasses.field(default=None, init=False)
    """The name of the algorithm."""

    num_batch: int = None
    """The dataset is divided into this number of groups where each group is
    processed sequentially."""

    batch_method: str = 'cluster_wobbly_center'
    """The name of the batch selection method. Choose from the cluster methods
    in the tike.random module."""

    costs: typing.List[typing.List[float]] = dataclasses.field(
        init=False,
        default_factory=list,
    )
    """The objective function value at previous iterations. One list is
    returned for each mini-batch."""

    num_iter: int = 1
    """The number of epochs to process before returning."""

    times: typing.List[float] = dataclasses.field(
        init=False,
        default_factory=list,
    )
    """The per-iteration wall-time for each previous iteration."""

    convergence_window: int = 0
    """The number of epochs to consider for convergence monitoring. Set to
    any value less than 2 to disable."""


@dataclasses.dataclass
class AdamOptions(IterativeOptions):
    name: str = dataclasses.field(default='adam_grad', init=False)

    alpha: float = 0.05
    """A hyper-parameter which controls the type of update regularization.
    RPIE becomes EPIE when this parameter is 1."""

    step_length: float = 1
    """Scales the search directions."""


@dataclasses.dataclass
class CgradOptions(IterativeOptions):
    name: str = dataclasses.field(default='cgrad', init=False)

    cg_iter: int = 2
    """The number of conjugate directions to search for each update."""

    step_length: float = 1
    """Scales the inital search directions before the line search."""


@dataclasses.dataclass
class DmOptions(IterativeOptions):
    name: str = dataclasses.field(default='dm', init=False)

    num_batch: int = 1
    """The dataset is divided into this number of groups where each group is
    processed simultaneously."""


@dataclasses.dataclass
class RpieOptions(IterativeOptions):
    name: str = dataclasses.field(default='rpie', init=False)

    num_batch: int = 5

    alpha: float = 0.05
    """A hyper-parameter which controls the step length. RPIE becomes EPIE when
    this parameter is 1."""


@dataclasses.dataclass
class LstsqOptions(IterativeOptions):
    name: str = dataclasses.field(default='lstsq_grad', init=False)


@dataclasses.dataclass
class PtychoParameters():
    """A class for storing the ptychography forward model parameters.

    .. versionadded:: 0.22.0
    """
    probe: np.array
    """(1, 1, SHARED, WIDE, HIGH) complex64 The shared illumination function
    amongst all positions."""

    psi: np.array
    """(WIDE, HIGH) complex64 The wavefront modulation coefficients of
    the object."""

    scan: np.array
    """(POSI, 2) float32 Coordinates of the minimum corner of the probe
    grid for each measurement in the coordinate system of psi. Coordinate order
    consistent with WIDE, HIGH order."""

    eigen_probe: np.array = None
    """(EIGEN, SHARED, WIDE, HIGH) complex64
    The eigen probes for all positions."""

    eigen_weights: np.array = None
    """(POSI, EIGEN, SHARED) float32
    The relative intensity of the eigen probes at each position."""

    algorithm_options: IterativeOptions = dataclasses.field(
        default_factory=RpieOptions,)
    """A class containing algorithm specific parameters"""

    probe_options: ProbeOptions = None
    """A class containing settings related to probe updates."""

    object_options: ObjectOptions = None
    """A class containing settings related to object updates."""

    position_options: PositionOptions = None
    """A class containing settings related to position correction."""

    def __post_init__(self):
        if (self.scan.ndim != 2 or self.scan.shape[1] != 2
                or np.any(np.asarray(self.scan.shape) < 1)):
            raise ValueError(f"scan shape {self.scan.shape} is incorrect. "
                             "It should be (N, 2) "
                             "where N >= 1 is the number of scan positions.")

        if (self.probe.ndim != 5 or self.probe.shape[:2] != (1, 1)
                or np.any(np.asarray(self.probe.shape) < 1)
                or self.probe.shape[-2] != self.probe.shape[-1]):
            raise ValueError(f"probe shape {self.probe.shape} is incorrect. "
                             "It should be (1, 1, S, W, H) "
                             "where S >=1 is the number of probes, and "
                             "W, H >= 1 are the square probe grid dimensions.")
        if (self.psi.ndim != 2 or np.any(
                np.asarray(self.psi.shape) <= np.asarray(self.probe.shape[-2:]))
           ):
            raise ValueError(
                f"psi shape {self.psi.shape} is incorrect. "
                "It should be (W, H) where W, H > probe.shape[-2:].")
        check_allowed_positions(
            self.scan,
            self.psi,
            self.probe.shape,
        )

    def resample(self, factor: float, interp):
        """Return a new PtychoParameter with the parameters scaled."""
        interp = _resize_fft if interp is None else interp
        return PtychoParameters(
            probe=interp(self.probe, factor),
            psi=_resize_spline(self.psi, factor),
            scan=self.scan * factor,
            eigen_probe=interp(self.eigen_probe, factor)
            if self.eigen_probe is not None else None,
            eigen_weights=self.eigen_weights,
            algorithm_options=self.algorithm_options,
            probe_options=self.probe_options.resample(factor)
            if self.probe_options is not None else None,
            object_options=self.object_options.resample(factor)
            if self.object_options is not None else None,
            position_options=self.position_options.resample(factor)
            if self.position_options is not None else None,
        )


def _resize_spline(x, f):
    return scipy.ndimage.zoom(
        x,
        zoom=[1] * (x.ndim - 2) + [f, f],
        grid_mode=True,
        prefilter=False,
    )


def _resize_cv(x, f, interpolation):
    import tike.view
    x_shape = x.shape
    x = x.reshape(-1, *x_shape[-2:])
    x1 = [
        tike.view.resize_complex_image(
            i,
            scale_factor=(f, f),
            interpolation=interpolation,
        ) for i in x
    ]
    return np.asarray(x1).reshape(*x_shape[:-2], *x1[0].shape[-2:])


def _resize_linear(x, f):
    return _resize_cv(x, f, 1)


def _resize_cubic(x, f):
    return _resize_cv(x, f, 2)


def _resize_lanczos(x, f):
    return _resize_cv(x, f, 4)


def crop_fourier_space(x, w: int):
    """Crop x assuming a 2D frequency space image with zero frequency in corner."""
    assert x.shape[-2] == x.shape[-1], "Only works on square arrays right now."
    half1 = w // 2
    half0 = w - half1
    # yapf: disable
    return x[
        ..., np.r_[0:half0, (x.shape[-1] - half1):x.shape[-1]],
    ][
        ..., np.r_[0:half0, (x.shape[-2] - half1):x.shape[-2]], :,
    ]
    # yapf: enable


def pad_fourier_space(x, w: int):
    """Pad x assuming a 2D frequency space image with zero frequency in corner."""
    assert x.shape[-2] == x.shape[-1], "Only works on square arrays right now."
    half1 = x.shape[-1] // 2
    half0 = x.shape[-1] - half1
    new_x = np.zeros_like(x, shape=(*x.shape[:-2], w, w))
    new_x[..., 0:half0, np.r_[0:half0, (w - half1):w]] = x[..., 0:half0, :]
    new_x[..., -half1:w, np.r_[0:half0, (w - half1):w]] = x[..., -half1:, :]
    return new_x


def _resize_fft(x, f):
    """Use Fourier interpolation to resize/resample the last 2 dimensions of x"""
    if f == 1:
        return x
    crop_or_pad = crop_fourier_space if f < 1 else pad_fourier_space
    return np.fft.ifft2(
        crop_or_pad(
            np.fft.fft2(
                x,
                norm='ortho',
                axes=(-2, -1),
            ),
            w=int(x.shape[-1] * f),
        ),
        norm='ortho',
        axes=(-2, -1),
    )
