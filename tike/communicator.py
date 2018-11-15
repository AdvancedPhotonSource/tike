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

    def get_ptycho_slice(self, tomo_slice):
        """Switch to slicing for the pytchography problem."""
        # Break the tomo data along the theta axis
        t_chunks = np.array_split(tomo_slice, self.size, axis=0)  # Theta, V, H
        # Each rank takes a turn scattering its tomo v slice to the others
        p_chunks = list()
        for i in range(self.size):
            p_chunks.append(self.comm.scatter(t_chunks, root=i))
        # Recombine the along vertical axis so each rank now has a theta slice
        return np.concatenate(p_chunks, axis=1)  # Theta, V, H

    def get_tomo_slice(self, ptych_slice):
        """Switch to slicing for the tomography problem."""
        # Break the ptych data along the vertical axis
        p_chunks = np.array_split(ptych_slice, self.size, axis=1)
        # Each rank takes a turn scattering its ptych theta slice to the others
        t_chunks = list()
        for i in range(self.size):
            t_chunks.append(self.comm.scatter(p_chunks, root=i))
        # Recombine along the theta axis so each rank now has a vertical slice
        return np.concatenate(t_chunks, axis=0)  # Theta, V, H

    def gather(self, arg, root=0, axis=0):
        arg = self.comm.gather(arg, root=root)  # Theta, V, H
        if self.rank == root:
            return np.concatenate(arg, axis=axis)
        return None
