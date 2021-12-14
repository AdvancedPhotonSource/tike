from abc import ABC
import dataclasses
from typing import List


@dataclasses.dataclass
class IterativeOptions(ABC):
    """

    Parameters
    ----------
    batch_size : int
        The approximate number of scan positions processed by each GPU
        simultaneously per view.
    costs : list[float]
        The objective function value at previous iterations
    num_iter : int
        The number of epochs to process before returning.
    times : list[float]
        The per-iteration wall-time for each previous iteration.

    .. versionadded:: 0.20.0
    """
    name: str = dataclasses.field(default=None, init=False)

    batch_size: int = None
    costs: List[float] = dataclasses.field(default_factory=list)
    num_iter: int = 1
    times: List[float] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class AdamOptions(IterativeOptions):
    name: str = dataclasses.field(default='adam_grad', init=False)


@dataclasses.dataclass
class CgradOptions(IterativeOptions):
    """

    Parameters
    ----------
    cg_iter : int
        The number of conjugate directions to search for each update.
    step_length : float
        Scales the inital search directions before the line search.
    """
    name: str = dataclasses.field(default='cgrad', init=False)

    cg_iter: int = 2
    step_length: float = 1


@dataclasses.dataclass
class EpieOptions(IterativeOptions):
    name: str = dataclasses.field(default='epie', init=False)

    batch_size: int = 1


@dataclasses.dataclass
class LstsqOptions(IterativeOptions):
    name: str = dataclasses.field(default='lstsq_grad', init=False)
