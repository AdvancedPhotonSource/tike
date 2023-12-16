import unittest

import numpy as np

import tike.ptycho

from .test_ptycho import PtychoRecon
from .templates import _mpi_size


@unittest.skipIf(
    True,
    reason="Data addition method broken until further notice.",
)
class TestPtychoOnline(PtychoRecon, unittest.TestCase):
    """Test ptychography reconstruction when data is streaming."""

    post_name = "-online"

    def setUp(self, chunks=16) -> None:
        """Modify the setup data to have streaming data."""
        PtychoRecon.setUp(self)
        data = np.array_split(self.data, chunks, axis=0)
        scan = np.array_split(self.scan, chunks, axis=0)
        assert len(data) == chunks
        assert len(scan) == chunks

        self.data = data[0]
        self.scan = scan[0]
        self.data_more = data[1:]
        self.scan_more = scan[1:]

    def template_consistent_algorithm(self, *, data, params):
        """Call tike.ptycho.Reconstruction with streaming data."""
        if self.mpi_size > 1:
            raise NotImplementedError()

        with tike.ptycho.Reconstruction(parameters=params,
                                        data=data) as context:
            context.iterate(2)
            for d, s in zip(self.data_more, self.scan_more):
                context.append_new_data(
                    new_data=d,
                    new_scan=s,
                )
                context.iterate(2)
        result = context.parameters
        print()
        print('\n'.join(f'{c[0]:1.3e}' for c in result.algorithm_options.costs))
        return result
