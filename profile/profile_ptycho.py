#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Benchmark ptychography reconstruction."""

import logging
import lzma
import os
import pickle
import unittest

import cupy as cp
import numpy as np
from pyinstrument import Profiler
import tike.ptycho


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

    def start(self):
        self.profiler.start()
        cp.cuda.profiler.start()

    def stop(self):
        cp.cuda.profiler.stop()
        self.profiler.stop()
        print('\n')
        print(self.profiler.output_text(
            unicode=True,
            color=True,
        ))

    @unittest.skip('Demonstrate skipped tests.')
    def test_never(self):
        """Never run this test."""
        pass

    def template_algorithm(self, algorithm):
        """Use pyinstrument to benchmark a ptycho algorithm on one core."""
        logging.disable(logging.WARNING)
        result = {
            'psi': np.ones_like(self.original),
            'probe': self.probe,
            'scan': self.scan,
        }
        # Do one iteration to complete JIT compilation
        result = tike.ptycho.reconstruct(
            **result,
            data=self.data,
            algorithm=algorithm,
            num_iter=1,
            rtol=-1,
        )
        self.start()
        result = tike.ptycho.reconstruct(
            **result,
            data=self.data,
            algorithm=algorithm,
            num_iter=100,
            rtol=-1,
        )
        self.stop()

    def test_combined(self):
        """Use pyinstrument to benchmark the combined algorithm."""
        self.template_algorithm('cgrad')

    def test_divided(self):
        """Use pyinstrument to benchmark the divided algorithm."""
        self.template_algorithm('lstsq_grad')


if __name__ == '__main__':
    unittest.main(verbosity=2)
