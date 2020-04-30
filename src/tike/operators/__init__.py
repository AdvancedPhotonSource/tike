"""Defines reference implementations for all operators.

All of the solvers, rely on core operators including forward and adjoint
operators. These core operators may have multiple implementations based on
different backends e.g. CUDA, OpenCL, NumPy. The classes in this module
prescribe an interface and reference implementation upon which specific solvers
are based. In this way, multiple solvers (e.g. ePIE, gradient descent, SIRT)
implemented in Python can share the same core operators and can be upgraded to
better operators in the future.

All operator methods should take NumPy arrays as inputs. This is a design
decision which was made because clients of the operators library should not
need to be concerned about memory locality which is necessary complexity when
implementating operators for specialized hardware (such as GPUs). Additionally,
we assume that GPU memory is too small to hold an entire workload at once.

"""

from .convolution import *
from .operator import *
from .propagation import *
from .ptycho import *
from .tomo import *

__all__ = (
    'Operator',
    'Convolution',
    'Propagation',
    'Ptycho',
    'Tomo',
)
