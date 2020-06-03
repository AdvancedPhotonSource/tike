from abc import ABC

import numpy
import cupy


class Operator(ABC):

    xp = cupy

    @classmethod
    def asarray(cls, *args, **kwargs):
        return cupy.asarray(*args, **kwargs)

    @classmethod
    def asnumpy(cls, *args, **kwargs):
        return cupy.asnumpy(*args, **kwargs)

    @classmethod
    def asarray_multi(cls, gpu_count, *args, **kwargs):
        mlist = [None] * gpu_count
        for i in range(gpu_count):
            with cupy.cuda.Device(i):
                mlist[i] = cupy.asarray(*args, **kwargs)
        return mlist

    @classmethod
    def asarray_multi_split(cls, gpu_count, scan_cpu, data_cpu,  *args, **kwargs):
        scanmlist = [None] * gpu_count
        datamlist = [None] * gpu_count
        nscan = scan_cpu.shape[1]
        tmplist = [0] * nscan
        counter = [0] * gpu_count
        xmax = numpy.amax(scan_cpu[:, :, 0])
        ymax = numpy.amax(scan_cpu[:, :, 1])
        for e in range(nscan):
            xgpuid = scan_cpu[0, e, 0] // (xmax/(gpu_count//2)) - int(scan_cpu[0, e, 0] != 0 and scan_cpu[0, e, 0] % (xmax/(gpu_count//2)) == 0)
            ygpuid = scan_cpu[0, e, 1] // (ymax/2) - int(scan_cpu[0, e, 1] != 0 and scan_cpu[0, e, 1] % (ymax/2) == 0)
            idx = int(xgpuid*2+ygpuid)
            tmplist[e] = idx
            counter[idx] += 1
        for i in range(gpu_count):
            tmpscan = numpy.zeros([scan_cpu.shape[0], counter[i], scan_cpu.shape[2]], dtype=scan_cpu.dtype)
            tmpdata = numpy.zeros([data_cpu.shape[0], counter[i], data_cpu.shape[2], data_cpu.shape[3]], dtype=data_cpu.dtype)
            c = 0
            for e in range(nscan):
                if tmplist[e] == i:
                    tmpscan[:, c, :] = scan_cpu[:, e, :]
                    tmpdata[:, c] = data_cpu[:, e]
                    c += 1
            with cupy.cuda.Device(i):
                scanmlist[i] = cupy.asarray(tmpscan)
                datamlist[i] = cupy.asarray(tmpdata)
            del tmpscan
            del tmpdata
        return scanmlist, datamlist
