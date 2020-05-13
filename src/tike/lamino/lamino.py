#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2018, UChicago Argonne, LLC. All rights reserved.    #
#                                                                         #
# Copyright 2018. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    "reconstruct",
    "simulate",
]

import logging
import numpy as np

from tike.operators import Lamino
from tike.lamino import solvers

logger = logging.getLogger(__name__)


def simulate(
        obj,
        theta,
        tilt,
        **kwargs
):  # yapf: disable
    """Return complex values of simulated laminography data."""
    assert obj.ndim == 3
    assert theta.ndim == 1
    with Lamino(
            n=obj.shape[-1],
            theta=theta,
            tilt=tilt,
            eps=1e-3,
            **kwargs,
    ) as operator:
        data = operator.fwd(obj)
        assert data.dtype == 'complex64', data.dtype
        data = operator.asnumpy(data)
        assert data.dtype == 'complex64', data.dtype

    return operator.asnumpy(data)


def reconstruct(
        data,
        theta,
        tilt,
        algorithm,
        obj=None, num_iter=1, rtol=-1, **kwargs
):  # yapf: disable
    """Solve the Laminography problem using the given `algorithm`.

    Parameters
    ----------
    algorithm : string
        The name of one algorithms from :py:mod:`.lamino.solvers`.
    rtol : float
        Terminate early if the relative decrease of the cost function is
        less than this amount.

    """
    n = data.shape[2]
    obj = np.zeros([n, n, n], dtype='complex64') if obj is None else obj
    if algorithm in solvers.__all__:
        # Initialize an operator.
        with Lamino(
            n=obj.shape[-1],
            theta=theta,
            tilt=tilt,
            eps=1e-3,
            **kwargs,
        ) as operator:
            # send any array-likes to device
            data = operator.asarray(data, dtype='complex64')
            result = {
                'obj': operator.asarray(obj, dtype='complex64'),
            }
            for key, value in kwargs.items():
                if np.ndim(value) > 0:
                    kwargs[key] = operator.asarray(value)

            logger.info("{} for {:,d} - {:,d} by {:,d} frames for {:,d} "
                        "iterations.".format(algorithm, *data.shape,
                                             num_iter))

            cost = 0
            for i in range(num_iter):
                kwargs.update(result)
                result = getattr(solvers, algorithm)(
                    operator,
                    data=data,
                    **kwargs,
                )
                # Check for early termination
                if i > 0 and abs((result['cost'] - cost) / cost) < rtol:
                    logger.info(
                        "Cost function rtol < %g reached at %d "
                        "iterations.", rtol, i)
                    break
                cost = result['cost']

        return {k: operator.asnumpy(v) for k, v in result.items()}
    else:
        raise ValueError(
            "The '{}' algorithm is not an available.".format(algorithm))
