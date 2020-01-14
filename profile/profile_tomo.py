#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Benchmark tomography reconstruction."""

import os
import logging
import lzma
import pickle
from pyinstrument import Profiler
import unittest
# These environmental variables must be set before numpy is imported anywhere.
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np  # noqa
import tike.tomo  # noqa


class BenchmarkTomo(unittest.TestCase):
    """Run benchmarks for tomography reconstruction."""

    def setUp(self):
        """Create a test dataset."""
        self.profiler = Profiler()
        dataset_file = '../tests/data/tomo_setup.pickle.lzma'
        with lzma.open(dataset_file, 'rb') as file:
            [
                self.data,
                self.theta,
                self.original,
            ] = pickle.load(file)

    @unittest.skip('Demonstrate skipped tests.')
    def test_never(self):
        """Never run this test."""
        pass

    def test_cgrad(self):
        """Use pyinstrument to benchmark tomo.grad on one core."""
        logging.disable(logging.WARNING)
        result = {
            'obj': np.zeros(self.original.shape, dtype=np.complex64)
        }
        self.profiler.start()
        for i in range(50):
            result = tike.tomo.reconstruct(
                **result,
                theta=self.theta,
                integrals=self.data,
                algorithm='cgrad',
                num_iter=1,
            )
        self.profiler.stop()
        print('\n')
        print(self.profiler.output_text(unicode=True, color=True))


if __name__ == '__main__':
    unittest.main(verbosity=2)
