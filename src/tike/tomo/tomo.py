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

__author__ = "Doga Gursoy, Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    "reconstruct",
    "simulate",
]
import logging

import numpy as np

from tike.tomo import TomoBackend
from tike.tomo.solvers import available_solvers

logger = logging.getLogger(__name__)


def reconstruct(
        obj,
        integrals,
        theta,
        algorithm=None, num_iter=1, **kwargs
):  # yapf: disable
    """Reconstruct the `obj` using the given `algorithm`."""
    xp = TomoBackend.array_module
    logger.info("{} on {:,d} - {:,d} by {:,d} grids for {:,d} "
                "iterations".format(algorithm, *integrals.shape,
                                    num_iter))
    original = xp.array(obj)
    if algorithm in available_solvers:
        solver = available_solvers[algorithm](
            ntheta=theta.size,
            nz=obj.shape[0],
            n=obj.shape[1],
            center=obj.shape[1] / 2,
        )
        result = solver.run(
            tomo=xp.asarray(integrals),
            obj=xp.asarray(obj),
            theta=xp.asarray(theta),
            num_iter=num_iter,
            **kwargs
        )  # yapf: disable
        return {
            'obj': TomoBackend.asnumpy(result['obj']),
        }
    else:
        raise ValueError(
            "The {} algorithm is not an available.".format(algorithm))


def simulate(
        obj,
        theta,
        **kwargs
):  # yapf: disable
    """Compute line integrals over an obj."""
    xp = TomoBackend.array_module
    assert obj.ndim == 3
    with TomoBackend(
        ntheta=theta.size,
        nz=obj.shape[0],
        n=obj.shape[1],
        center=obj.shape[1] / 2,
    ) as slv:
        integrals = slv.fwd(
            obj=xp.asarray(obj),
            theta=xp.array(theta),
        )
    return TomoBackend.asnumpy(integrals)
