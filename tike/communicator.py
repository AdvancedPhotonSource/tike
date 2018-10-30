"""Define an communication class to move data between processes."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import numpy as np
from mpi4py import MPI

__author__ = "Doga Gursoy, Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = []

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MPICommunicator(object):
    """Communicate between processes using MPI.

    Use this class to astract away all of the MPI communication that needs to
    occur in order to switch between the tomography and ptychography problems.
    """

    def __init__(self):
        """Load the MPI params and get initial data."""
        super(MPICommunicator, self).__init__()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        logger.info("Node {:,d} is running.".format(self.rank))

    def scatter(self, *args):
        """Send and recieve constant data that must be divided."""
        out = list()
        for arg in args:
            if self.rank == 0:
                chunks = np.array_split(arg, self.size)
            else:
                chunks = None
            out.append(self.comm.scatter(chunks, root=0))
        return out

    def broadcast(self, *args):
        """Synchronize parameters that are the same for all processses."""
        out = list()
        for arg in args:
            out.append(self.comm.bcast(arg, root=0))
        return out

    def get_ptycho_slice(self, arg):
        """Switch to arg slicing for the pytchography problem."""
        arg = self.comm.allgather(arg)  # Theta, V, H
        whole = np.concatenate(arg, axis=1)
        return np.array_split(whole, self.size, axis=0)[self.rank]

    def get_tomo_slice(self, arg):
        """Switch to arg slicing for the tomography problem."""
        arg = self.comm.allgather(arg)  # Theta, V, H
        whole = np.concatenate(arg, axis=0)
        return np.array_split(whole, self.size, axis=1)[self.rank]

    def gather(self, arg, root=0, axis=0):
        arg = self.comm.allgather(arg)  # Theta, V, H
        return np.concatenate(arg, axis=axis)

