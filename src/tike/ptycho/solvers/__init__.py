"""Contains different solver implementations."""

from .adam import adam_grad
from .conjugate import cgrad
from .lstsq import lstsq_grad
from .epie import epie

__all__ = [
    'adam_grad',
    'cgrad',
    'lstsq_grad',
    'epie',
]
