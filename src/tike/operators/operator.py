from abc import ABC

import numpy

class Operator(ABC):
    """A base class for Operators.

    An Operator is a context manager which provides the basic functions (forward
    and adjoint) required solve an inverse problem.

    Attributes
    ----------
    array_module : Python module
        A module which provides a NumPy interface that will be compatible with
        the arrays passed to the operator.
    asnumpy : function
        This function converts the arrays of the type processed by this
        operator into NumPy arrays.

    Attributes
    ----------
    xp : module
        Provides the array functions that this operator uses i.e. NumPy, Cupy
    asnumpy : function
        Converts this operators native array to a numpy array.

    """
    xp = numpy

    @classmethod
    def asarray(cls, *args, **kwargs):
        return numpy.asarray(*args, **kwargs)

    @classmethod
    def asnumpy(cls, *args, **kwargs):
        return numpy.asarray(*args, **kwargs)

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Gracefully handle interruptions or with-block exit.

        Tasks to be handled by this function include freeing memory or closing
        files.
        """
        pass

    def fwd(self, **kwargs):
        """Perform the forward operator."""
        raise NotImplementedError("The forward operator was not implemented!")

    def adj(self, **kwargs):
        """Perform the adjoint operator."""
        raise NotImplementedError("The adjoint operator was not implemented!")
