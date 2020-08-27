#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test functions in tike.trajectory."""

import numpy as np
from tike.trajectory import *

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


def test_discrete_trajectory():
    """Check trajectory.discrete_trajectory for stationary probe."""

    def stationary(t):
        """Probe is stationary at location h = 8, v = 8."""
        return 0 * t, 8 + 0 * t, 8 + 0 * t

    answer = discrete_trajectory(
        stationary,
        tmin=0,
        tmax=0.65,
        xstep=0.1,
        tstep=1,
    )
    truth = ([0], [8], [8], [0.65], [0])
    np.testing.assert_equal(answer, truth)


def test_coded_exposure():
    """Check trajectory.coded_exposure for correctness."""
    c_time = np.arange(11)
    c_dwell = np.ones(11) * 0.5

    time = np.array(
        [-1., 0.8, 1.8, 3.0, 4.1, 4.2, 6.1, 7.5, 8.6, 8.9, 8.9, 8.9, 20, 21])
    dwell = np.array(
        [0.1, 0.2, 0.4, 0.5, 0.1, 0.1, 0.6, 0.2, 0.2, 2, 0, 0.3, 1.0, 1.0])

    theta = np.arange(time.size)
    v = np.arange(time.size)
    h = np.arange(time.size)

    th1, v1, h1, t1, d1, b1 = coded_exposure(theta, v, h, time, dwell, c_time,
                                             c_dwell)

    np.testing.assert_equal(th1, [2, 3, 4, 5, 6, 9, 11, 9])
    np.testing.assert_equal(v1, [2, 3, 4, 5, 6, 9, 11, 9])
    np.testing.assert_equal(h1, [2, 3, 4, 5, 6, 9, 11, 9])
    np.testing.assert_equal(t1, [2., 3., 4.1, 4.2, 6.1, 9., 9., 10.])
    np.testing.assert_allclose(d1, [0.2, 0.5, 0.1, 0.1, 0.4, 0.5, 0.2, 0.5])
    np.testing.assert_equal(b1, [0, 1, 2, 4, 5, 7])


if __name__ == '__main__':
    test_discrete_trajectory()
    test_coded_exposure()
