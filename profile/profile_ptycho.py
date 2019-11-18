#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Benchmark ptychography reconstruction."""

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
import tike.ptycho  # noqa


class BenchmarkPtycho(unittest.TestCase):
    """Run benchmarks for pychography reconstruction."""

    def setUp(self):
        """Create a test dataset."""
        self.profiler = Profiler()
        dataset_file = '../tests/data/ptycho_setup.pickle.lzma'
        with lzma.open(dataset_file, 'rb') as file:
            [
                self.data,
                self.scan,
                self.probe,
                self.original,
            ] = pickle.load(file)

    @unittest.skip('Demonstrate skipped tests.')
    def test_never(self):
        """Never run this test."""
        pass

    def test_cgrad(self):
        """Use pyinstrument to benchmark ptycho.grad on one core."""
        logging.disable(logging.WARNING)
        result = {
            'psi': np.ones_like(self.original),
            'probe': self.probe,
        }
        self.profiler.start()
        for i in range(50):
            result = tike.ptycho.reconstruct(
                **result,
                data=self.data,
                scan=self.scan,
                algorithm='cgrad',
                num_iter=1,
                rho=0,
                gamma=0.5
                )
        self.profiler.stop()
        print('\n')
        print(self.profiler.output_text(unicode=True, color=True))


if __name__ == '__main__':
    unittest.main(verbosity=2)
