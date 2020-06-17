from abc import ABC

import numpy
import cupy


class Operator(ABC):

    xp = cupy

    @classmethod
    def asarray(cls, *args, device=None, **kwargs):
        with cupy.cuda.Device(device):
            return cupy.asarray(*args, **kwargs)

    @classmethod
    def asnumpy(cls, *args, **kwargs):
        return cupy.asnumpy(*args, **kwargs)

    @classmethod
    def asarray_multi(cls, gpu_count, *args, **kwargs):
        return [
            cls.asarray(*args, device=i, **kwargs) for i in range(gpu_count)
        ]

    @classmethod
    def asarray_multi_split(cls, gpu_count, scan, data, fly=1):
        """Split scan and data and distribute to multiple GPUs.

        Divide the work amongst multiple GPUS by splitting the field of view
        along the vertical axis. i.e. each GPU gets a subset of the data that
        correspond to a horizontal stripe of the field of view.

        This type of division does not minimze the volume of data transferred
        between GPUs. However, it does minimizes the number of neighbors that
        each GPU must communicate with, supports odd numbers of GPUs, and makes
        resizing the subsets easy if the scan positions are not evenly
        distributed across the field of view.

        FIXME: Only uses the first angle to divide the positions. Assumes the
        positions on all angles are distributed similarly.

        Parameters
        ----------
        scan (ntheta, nscan, 2) float32
            Scan positions.
        data (ntheta, nscan // fly, D, D) float32
            Captured frames from the detector.
        fly : int
            The number of scan positions per data frame
        """
        # Reshape scan so positions in the same fly scan are not separated
        ntheta, nscan, _ = scan.shape
        scan = scan.reshape(ntheta, nscan // fly, fly, 2)
        # Determine the edges of the horizontal stripes
        edges = numpy.linspace(
            0,
            scan[..., 0].max(),
            gpu_count + 1,
            endpoint=True,
        )
        # Split the scan positions and data amongst the stripes
        scanmlist = []
        datamlist = []
        for i in range(gpu_count):
            keep = numpy.logical_and(
                edges[i] < scan[0, :, 0, 0],
                scan[0, :, 0, 0] <= edges[i + 1],
            )
            scanmlist.append(scan[:, keep].reshape(ntheta, -1, 2))
            datamlist.append(data[:, keep])
        # Send each chunk to a GPU
        for i in range(gpu_count):
            scanmlist[i] = cls.asarray(scanmlist[i], device=i)
            datamlist[i] = cls.asarray(datamlist[i], device=i)

        return scanmlist, datamlist

    @classmethod
    def asarray_multi_fuse(cls, gpu_count, *args, **kwargs):
        """Collect and fuse the data into one GPU.

        Each element of the args, e.g., args[0] is expected to be a list of
        cupy array. The size of each list is the same as gpu_count.

        """
        with cupy.cuda.Device(0):
            fused = args[0][0].copy()
        for i in range(1, gpu_count):
            with cupy.cuda.Device(i):
                fused_cpu = cupy.asnumpy(args[0][i])
                with cupy.cuda.Device(0):
                    fused = cupy.concatenate(
                        (fused, cupy.asarray(fused_cpu)),
                        axis=1,
                    )
        return fused
