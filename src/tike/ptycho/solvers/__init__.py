"""Contains different solver implementations."""

from .adam import adam_grad
from .combined import cgrad
from .divided import lstsq_grad

__all__ = [
    'adam_grad',
    'cgrad',
    'lstsq_grad',
]
