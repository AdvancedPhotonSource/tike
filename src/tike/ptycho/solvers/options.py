import abc
import dataclasses
import typing

import numpy as np

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

    costs: typing.List[float] = dataclasses.field(
        init=False,
        default_factory=list,
    )
    """The objective function value at previous iterations."""

    num_iter: int = 1
    """The number of epochs to process before returning."""

    times: typing.List[float] = dataclasses.field(
        init=False,
        default_factory=list,
    )
    """The per-iteration wall-time for each previous iteration."""


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
