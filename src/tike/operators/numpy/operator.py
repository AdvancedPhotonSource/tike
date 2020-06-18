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
    def asarray(cls, *args, device=None, **kwargs):
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

    @classmethod
    def asarray_multi(cls, gpu_count, *args, **kwargs):
        return [
            cls.asarray(*args, device=i, **kwargs) for i in range(gpu_count)
        ]

    @classmethod
    def asarray_multi_split(cls, gpu_count, scan_cpu, data_cpu, *args,
                            **kwargs):
        """Split scan and data and distribute to multiple GPUs.

        Instead of spliting the arrays based on the scanning order, we split
        them in accordance with the scan positions corresponding to the object
        sub-images. For example, if we divide a square object image into four
        sub-images, then the scan positions on the top-left sub-image and their
        corresponding diffraction patterns will be grouped into the first chunk
        of scan and data.

        """
        scanmlist = [None] * gpu_count
        datamlist = [None] * gpu_count
        nscan = scan_cpu.shape[1]
        tmplist = [0] * nscan
        counter = [0] * gpu_count
        xmax = numpy.amax(scan_cpu[:, :, 0])
        ymax = numpy.amax(scan_cpu[:, :, 1])
        for e in range(nscan):
            xgpuid = scan_cpu[0, e, 0] // (xmax / (gpu_count // 2)) - int(
                scan_cpu[0, e, 0] != 0 and scan_cpu[0, e, 0] %
                (xmax / (gpu_count // 2)) == 0)
            ygpuid = scan_cpu[0, e, 1] // (ymax / 2) - int(
                scan_cpu[0, e, 1] != 0 and scan_cpu[0, e, 1] % (ymax / 2) == 0)
            idx = int(xgpuid * 2 + ygpuid)
            tmplist[e] = idx
            counter[idx] += 1
        for i in range(gpu_count):
            tmpscan = numpy.zeros(
                [scan_cpu.shape[0], counter[i], scan_cpu.shape[2]],
                dtype=scan_cpu.dtype,
            )
            tmpdata = numpy.zeros(
                [
                    data_cpu.shape[0], counter[i], data_cpu.shape[2],
                    data_cpu.shape[3]
                ],
                dtype=data_cpu.dtype,
            )
            c = 0
            for e in range(nscan):
                if tmplist[e] == i:
                    tmpscan[:, c, :] = scan_cpu[:, e, :]
                    tmpdata[:, c] = data_cpu[:, e]
                    c += 1
                scanmlist[i] = cls.asarray(tmpscan, device=i)
                datamlist[i] = cls.asarray(tmpdata, device=i)
            del tmpscan
            del tmpdata
        return scanmlist, datamlist

    @classmethod
    def asarray_multi_fuse(cls, gpu_count, *args, **kwargs):
        """Collect and fuse the data into one GPU.

        Each element of the args, e.g., args[0] is expected to be a list of
        cupy array. The size of each list is the same as gpu_count.

        """
        fused = args[0][0].copy()
        for i in range(1, gpu_count):
            fused_cpu = cls.asnumpy(args[0][i])
            fused = cls.xp.concatenate(
                (fused, cls.asarray(fused_cpu, device=0)),
                axis=1,
            )
        return fused
