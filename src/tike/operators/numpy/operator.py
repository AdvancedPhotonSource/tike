from abc import ABC

import numpy


class Operator(ABC):
    """A base class for Operators.

    An Operator is a context manager which provides the basic functions
    (forward and adjoint) required solve an inverse problem.

    Operators may be composed into other operators and inherited from to
    provide additional implementations to the ones provided in this library.

    """
    xp = numpy
    """The module of the array type used by this operator i.e. NumPy, Cupy."""

    @classmethod
    def asarray(cls, *args, **kwargs):
        """Convert NumPy arrays into the array-type of this operator."""
        return numpy.asarray(*args, **kwargs)

    @classmethod
    def asnumpy(cls, *args, **kwargs):
        """Convert the arrays of this operator into NumPy arrays."""
        return numpy.asarray(*args, **kwargs)

    def __enter__(self):
        """Return self at start of a with-block."""
        # Call the __enter__ methods for any composed operators.
        # Allocate special memory objects.
        return self

    def __exit__(self, type, value, traceback):
        """Gracefully handle interruptions or with-block exit.

        Tasks to be handled by this function include freeing memory or closing
        files.
        """
        # Call the __exit__ methods of any composed classes.
        # Deallocate special memory objects.
        pass

    def fwd(self, **kwargs):
        """Perform the forward operator."""
        raise NotImplementedError("The forward operator was not implemented!")

    def adj(self, **kwargs):
        """Perform the adjoint operator."""
        raise NotImplementedError("The adjoint operator was not implemented!")
