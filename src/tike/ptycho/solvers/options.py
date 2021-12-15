from abc import ABC
import dataclasses
from typing import List


@dataclasses.dataclass
class IterativeOptions(ABC):
    """A base class providing options for iterative algorithms.

    .. versionadded:: 0.20.0
    """
    name: str = dataclasses.field(default=None, init=False)
    """The name of the algorithm."""

    batch_size: int = None
    """The approximate number of scan positions processed by each GPU
    simultaneously per view."""

    costs: List[float] = dataclasses.field(default_factory=list)
    """The objective function value at previous iterations."""

    num_iter: int = 1
    """The number of epochs to process before returning."""

    times: List[float] = dataclasses.field(default_factory=list)
    """The per-iteration wall-time for each previous iteration."""


@dataclasses.dataclass
class AdamOptions(IterativeOptions):
    name: str = dataclasses.field(default='adam_grad', init=False)


@dataclasses.dataclass
class CgradOptions(IterativeOptions):
    name: str = dataclasses.field(default='cgrad', init=False)

    cg_iter: int = 2
    """The number of conjugate directions to search for each update."""

    step_length: float = 1
    """Scales the inital search directions before the line search."""


@dataclasses.dataclass
class EpieOptions(IterativeOptions):
    name: str = dataclasses.field(default='epie', init=False)

    batch_size: int = 1


@dataclasses.dataclass
class LstsqOptions(IterativeOptions):
    name: str = dataclasses.field(default='lstsq_grad', init=False)
