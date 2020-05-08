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

from tike.operators import numpy as default
default_backend = "numpy"


def _set_backend(requested_backend):
    """Set the operators to the requested_backend.

    Try loading all of the entry points. If the requested_backend fails,
    provide the reason why and show the backends that did not fail.
    """
    for operator in ['Operator', 'Ptycho', 'Convolution', 'Propagation', 'Lamino']:
        if requested_backend == default_backend:
            globals()[operator] = getattr(default, operator)
            continue

        backend_options = {}
        failed_import = []
        for entry_point in pkg_resources.iter_entry_points(f'tike.{operator}'):
            try:
                backend_options[entry_point.name] = entry_point.load()
            except ImportError as error:
                failed_import.append(f"{entry_point.name}: {error}")
        if requested_backend in backend_options:
            globals()[operator] = backend_options[requested_backend]
        else:
            raise ImportError(
                f"Cannot set {operator} operator as '{requested_backend}'. "
                f"Available backends: {list(backend_options.keys())}. "
                f"Unavailable backends: {failed_import}.")


# Search available entry points for requested backend.
if f"TIKE_BACKEND" in os.environ:
    _set_backend(os.environ["TIKE_BACKEND"])
else:
    _set_backend(default_backend)
