"""Defines implementations for all operators.

All of the solvers, rely on operators including forward and adjoint operators.
In tike, forward and adjoint operators are paired as fwd and adj methods of an
Operator.

In this way, multiple solvers (e.g. ePIE, gradient descent, SIRT) implemented
in Python can share the same core operators and can be upgraded to better
operators in the future.

All operator methods accept the array type that matches the output of their
asarray() method.
"""

from .cupy import *
