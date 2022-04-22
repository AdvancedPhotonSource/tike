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

    num_batch: int = None
    """The dataset is divided into this number of groups where each group is
    processed sequentially."""

    batch_method: str = 'cluster_wobbly_center'
    """The name of the batch selection method. Choose from the cluster_ methods
    in the tike.random module."""

    costs: List[float] = dataclasses.field(init=False, default_factory=list)
    """The objective function value at previous iterations."""

    num_iter: int = 1
    """The number of epochs to process before returning."""

    times: List[float] = dataclasses.field(init=False, default_factory=list)
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
