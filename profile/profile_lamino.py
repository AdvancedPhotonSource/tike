#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Benchmark ptychography reconstruction."""

import logging
import lzma
import os
import pickle
from pyinstrument import Profiler
import unittest

# These environmental variables must be set before numpy is imported anywhere.
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np  # noqa
import tike.lamino  # noqa


class BenchmarkPtycho(unittest.TestCase):
    """Run benchmarks for laminography reconstruction."""

    def setUp(self):
        self.profiler = Profiler()
        dataset_file = '../tests/data/lamino_setup.pickle.lzma'
        with lzma.open(dataset_file, 'rb') as file:
            [
                self.data,
                self.original,
                self.theta,
                self.tilt,
            ] = pickle.load(file)

    def template_algorithm(self, algorithm):
        result = {
            'obj': np.zeros_like(self.original),
        }
        self.profiler.start()
        result = tike.lamino.reconstruct(
            **result,
            data=self.data,
            theta=self.theta,
            tilt=self.tilt,
            algorithm=algorithm,
            num_iter=10,
        )
        self.profiler.stop()
        print('\n')
        print(self.profiler.output_text(
            unicode=True,
            color=True,
        ))

    def test_cgrad(self):
        """Use pyinstrument to benchmark the conjugate gradient algorithm."""
        self.template_algorithm('cgrad')


if __name__ == '__main__':
    unittest.main(verbosity=2)
