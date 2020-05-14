from .. import numpy
from .convolution import Convolution
from .propagation import Propagation
from .operator import Operator
import concurrent.futures as cf
import threading
from functools import partial
import cupy as cp
import numpy as np


class Ptycho(Operator, numpy.Ptycho):
    def __init__(self, *args, **kwargs):
        super(Ptycho, self).__init__(
            *args,
            propagation=Propagation,
            diffraction=Convolution,
            **kwargs,
        )

    def fwd_multi(self, gpu_count, probe, scan, psi, **kwargs):
        self.diffraction.fwd_multi(gpu_count, psi, scan, probe)
        exit()
        #gpu_count = gpu_id
        gpu_list = range(gpu_count)
        #def multiGPU_init(gpu_i):
        #    print('tst:', gpu_i)
        with cf.ThreadPoolExecutor(max_workers=gpu_count) as executor:
            intensity = executor.map(self.diffraction.fwd_multi, gpu_list, psi, scan, probe)

        exit()
        a= self.diffraction.fwd_multi(
                gpu_count=gpu_count,
                psi=psi,
                scan=scan,
                probe=probe,
            )
        a = self.propagation.fwd_multi(
            gpu_id,
            self.diffraction.fwd_multi(
                gpu_count=gpu_count,
                psi=psi,
                scan=scan,
                probe=probe,
            ),
            overwrite=True,
        )
        print('inten:', scan.shape, scan[:, :, 1])
        return scan

    def _compute_intensity_multi(self, gpu_id, data, psi, scan, probe, n=-1, mode=None):
        """Compute detector intensities replacing the nth probe mode"""

        fwd_out = self.fwd_multi(
                            gpu_count=gpu_id,
                            psi=psi,
                            scan=scan,
                            probe=probe,
                        )
        exit()
        gpu_count = gpu_id
        intensity = [0] * gpu_count
        for i in range(gpu_count):
            with cp.cuda.Device(i):
                for m in range(probe[i].shape[-3]):
                    intensity[i] += cp.sum(
                        cp.square(cp.abs(self.fwd_multi(
                            psi=psi[i],
                            scan=scan[i],
                            probe=mode if m == n else probe[..., m:m + 1, :, :],
                        ).reshape(*data[i].shape[:2], -1, *data[i].shape[2:]))),
                        axis=2,
                    )  # yapf: disable

            intensity += np.sum(
                np.square(np.abs(self.fwd_multi(
                    gpu_id=gpu_id,
                    psi=psi,
                    scan=scan,
                    probe=probe,
                )))
            )  # yapf: disable
        return intensity

    def cost_device(self, gpu_id, data, psi, scan, probe, n=-1, mode=None): # cupy arrays
        with cp.cuda.Device(gpu_id):
            intensity = self._compute_intensity(data, psi, scan, probe, n, mode)
            return self.propagation.cost(data, intensity)

    # multi-GPU cost() entry point
    def cost_multi(self, gpu_count, data, psi, scan, probe, n=-1, mode=None, **kwargs):  # lists of cupy array
        psi_list = [None] * gpu_count
        for i in range(gpu_count):
            if 'step_length' in kwargs and 'dir' in kwargs:
                with cp.cuda.Device(i):
                    #print('testpsii:', i, psi[i].tolist())
                    #print('testdiri:', i,(kwargs.get('step_length')*kwargs.get('dir')[i]).tolist())
                    psi_list[i] = psi[i] + kwargs.get('step_length') * kwargs.get('dir')[i]
            else:
                psi_list[i] = psi[i]

        #for i in range(gpu_count):
        #    with cp.cuda.Device(i):
        #        print('testpsi:', i, psi[i].tolist())

        gpu_list = range(gpu_count)
        with cf.ThreadPoolExecutor(max_workers=gpu_count) as executor:
            cost_out = executor.map(self.cost_device, gpu_list, data, psi_list, scan, probe)
        cost_list = list(cost_out)

        cost_cpu = np.zeros(cost_list[0].shape, cost_list[0].dtype)
        for i in range(gpu_count):
            with cp.cuda.Device(i):
                cost_cpu += Operator.asnumpy(cost_list[i])
        print('test:', type(cost_cpu), cost_cpu.dtype, cost_cpu.shape, cost_cpu)

        return cost_cpu

    def grad_device(self, gpu_id, data, psi, scan, probe):  # cupy arrays
        with cp.cuda.Device(gpu_id):
            intensity = self._compute_intensity(data, psi, scan, probe)
            grad_obj = self.xp.zeros_like(psi)
            for mode in cp.split(probe, probe.shape[-3], axis=-3):
                # TODO: Pass obj through adj() instead of making new obj inside
                grad_obj += self.adj(
                    farplane=self.propagation.grad(
                        data,
                        self.fwd(psi=psi, scan=scan, probe=mode),
                        intensity,
                    ),
                    probe=mode,
                    scan=scan,
                    overwrite=True,
                )
            return grad_obj

    # multi-GPU grad() entry point
    def grad_multi(self, gpu_count, data, psi, scan, probe):  # lists of cupy array
        #intensity = self._compute_intensity_multi(gpu_id, data, psi, scan, probe) # intensity is a list of cupy arrays
        #gpu_count = gpu_id
        gpu_list = range(gpu_count)
        #def multiGPU_init(gpu_i):
        #    print('tst:', gpu_i)
        with cf.ThreadPoolExecutor(max_workers=gpu_count) as executor:
            grad_out = executor.map(self.grad_device, gpu_list, data, psi, scan, probe)
        grad_list = list(grad_out)

        with cp.cuda.Device(0):
            grad_tmp = cp.empty(grad_list[0].shape, grad_list[0].dtype)
            for i in range(1, gpu_count):
                if cp.cuda.runtime.deviceCanAccessPeer(0, i):
                    cp.cuda.runtime.deviceEnablePeerAccess(i)
                    grad_tmp.data.copy_from_device(grad_list[i].data, grad_list[0].size*grad_list[0].itemsize)
                else:
                    with cp.cuda.Device(i):
                        grad_cpu_tmp = Operator.asnumpy(grad_list[i])
                    grad_tmp = Operator.asarray(grad_cpu_tmp)
                grad_list[0] += grad_tmp

        return grad_list[0]
        #print('testgrad2:', grad_tmp.shape)
        #print('testgrad:', grad_list[0].tolist())
        #print('testgrad1:', grad_list[1].tolist())




        #with cp.cuda.Device(gpu_id):
        #    #print('test1:', gpu_id, scan[0][:, :, 0], scan[0][:, :, 1])
        #    #print('test1:', gpu_id, psi[0].shape) print('test2:', gpu_id, self.nscan)
        #    intensity = self._compute_intensity_multi(gpu_id, data[gpu_id], psi[gpu_id], scan[gpu_id], probe[gpu_id])
        #    print('test1:', gpu_id, intensity)
        #    grad_obj = self.xp.zeros_like(psi[gpu_id])
        #    print('data:', gpu_id, scan[gpu_id][:, :, 1])
        #    #for mode in cp.split(probe[gpu_id], probe[gpu_id].shape[-3], axis=-3):
        #    #    # TODO: Pass obj through adj() instead of making new obj inside
        #    #    grad_obj += self.adj(
        #    #        farplane=self.propagation.grad(
        #    #            data[gpu_id],
        #    #            self.fwd(psi=psi, scan=scan[gpu_id], probe=mode),
        #    #            intensity,
        #    #        ),
        #    #        probe=mode,
        #    #        scan=scan[gpu_id],
        #    #        overwrite=True,
        #    #    )
        #    print('data:')
        #        #print('grad_obj:', grad_obj.shape)
        #    return grad_obj
