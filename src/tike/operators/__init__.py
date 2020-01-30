"""This module defines reference implementations for all operators.

All of the solvers, rely on core operators including forward and adjoint
operators. These core operators may have multiple implementations based on
different backends e.g. CUDA, OpenCL, NumPy. The classes in this module
prescribe an interface and reference implementation upon which specific solvers
are based. In this way, multiple solvers (e.g. ePIE, gradient descent, SIRT)
implemented in Python can share the same core operators and can be upgraded to
better operators in the future.

"""

from .convolution import *
from .operator import *
from .propagation import *
from .ptycho import *
from .tomo import *
