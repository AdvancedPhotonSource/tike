"""Contains different solver implementations."""

from .combined import cgrad
from .divided import lstsq_grad

__all__ = [
    'cgrad',
    'lstsq_grad',
]
