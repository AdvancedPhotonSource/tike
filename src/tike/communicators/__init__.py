"""Module for communicators using threadpool and MPI.

This module implements both the p2p and collective communications
among multiple GPUs and multiple nodes.

"""

from .mpi import *
from .pool import *
from .comm import *

__all__ = (
    'ThreadPool',
    'MPIComm',
    'Comm',
)
