"""Defines implementations for all operators.

All of the solvers, rely on operators including forward and adjoint operators.
In tike, forward and adjoint operators are paired as fwd and adj methods of an
Operator.

These Operators may have multiple implementations based on different libraries
e.g. CUDA, OpenCL, NumPy. The classes in the operators.numpy module prescribe
an interface and reference implementation upon which other operators are based.
In this way, multiple solvers (e.g. ePIE, gradient descent, SIRT) implemented
in Python can share the same core operators and can be upgraded to better
operators in the future. Operator implementations are selected by setting the
TIKE_BACKEND environment variable.

All operator methods accept the array type that matches the output of their
asarray() method.
"""

import os
import pkg_resources
import warnings


def _set_operators(requested_backend):
    """Set the operators from the requested_backend.
    
    requested_backend is a python module which has some of the operators
    implemented.
    """
    module_attributes = dir(requested_backend)
    for operator in [
            'Convolution',
            'Flow',
            'Lamino',
            'Operator',
            'Propagation',
            'Ptycho',
            'Shift',
    ]:
        if operator in module_attributes:
            globals()[operator] = getattr(requested_backend, operator)
        else:
            warnings.warn(
                f"The {operator} operator is not implemented in "
                f"'{requested_backend}'.", ImportWarning)


_backend_options = {}
for backend in pkg_resources.iter_entry_points(f'tike.operators'):
    _backend_options[backend.name] = backend.load()

requested_backend = 'numpy'
if "TIKE_BACKEND" in os.environ:
    requested_backend = os.environ["TIKE_BACKEND"]

if requested_backend in _backend_options:
    _set_operators(_backend_options[requested_backend])
else:
    raise ImportError(f"Cannot set backend as '{requested_backend}'. "
                      f"Available backends: {list(_backend_options.keys())}.")
