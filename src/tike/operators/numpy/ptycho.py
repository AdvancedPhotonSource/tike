"""Defines a ptychography operator based on the NumPy FFT module."""

import concurrent.futures as cf

import numpy as np

from .operator import Operator
from .propagation import Propagation
from .convolution import Convolution


class Ptycho(Operator):
    """A Ptychography operator.

    Compose a diffraction and propagation operator to simulate the interaction
    of an illumination wavefront with an object followed by the propagation of
    the wavefront to a detector plane.

    Attributes
    ----------
    nscan : int
        The number of scan positions at each angular view.
    fly : int
        The number of consecutive scan positions that describe a fly scan.
    probe_shape : int
        The pixel width and height of the (square) probe illumination.
    detector_shape : int
        The pixel width and height of the (square) detector grid.
    nz, n : int
        The pixel width and height of the reconstructed grid.
    ntheta : int
        The number of angular partitions of the data.
    model : string
        The type of noise model to use for the cost functions.
    propagation : Operator
        The wave propagation operator being used.
    diffraction : Operator
        The object probe interaction operator being used.

    Parameters
    ----------
    psi : (ntheta, nz, n) complex64
        The complex wavefront modulation of the object.
    probe : complex64
        The complex (ntheta, nscan // fly, fly, 1, probe_shape,
        probe_shape) illumination function.
    mode : complex64
        A single (ntheta, nscan // fly, fly, 1, probe_shape, probe_shape)
        probe mode.
    nearplane, farplane: complex64
        The (ntheta, nscan // fly, fly, 1, detector_shape, detector_shape)
        wavefronts exiting the object and hitting the detector respectively.
    data : (ntheta, nscan // fly, detector_shape, detector_shape) float32
        The square of the absolute value of `farplane` summed over `fly` and
        `modes`.
    scan : (ntheta, nscan, 2) float32
        Coordinates of the minimum corner of the probe grid for each
        measurement in the coordinate system of psi. Vertical coordinates
        first, horizontal coordinates second.

    """

    def __init__(self, detector_shape, probe_shape, nz, n,
                 ntheta=1, model='gaussian', fly=1,
                 propagation=Propagation,
                 diffraction=Convolution,
                 **kwargs):  # noqa: D102 yapf: disable
        """Please see help(Ptycho) for more info."""
        self.propagation = propagation(
            detector_shape=detector_shape,
            model=model,
            **kwargs,
        )
        self.diffraction = diffraction(
            probe_shape=probe_shape,
            detector_shape=detector_shape,
            nz=nz,
            n=n,
            ntheta=ntheta,
            model=model,
            fly=fly,
            **kwargs,
        )
        # TODO: Replace these with @property functions
        self.probe_shape = probe_shape
        self.detector_shape = detector_shape
        self.nz = nz
        self.n = n
        self.ntheta = ntheta
        self.fly = fly

    def __enter__(self):
        self.propagation.__enter__()
        self.diffraction.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self.propagation.__exit__(type, value, traceback)
        self.diffraction.__exit__(type, value, traceback)

    def fwd(self, probe, scan, psi, **kwargs):
        return self.propagation.fwd(
            self.diffraction.fwd(
                psi=psi,
                scan=scan,
                probe=probe,
            ),
            overwrite=True,
        )

    def adj(self, farplane, probe, scan, overwrite=False, **kwargs):
        return self.diffraction.adj(
            nearplane=self.propagation.adj(
                farplane,
                overwrite=overwrite,
            ),
            probe=probe,
            scan=scan,
            overwrite=True,
        )

    def adj_probe(self, farplane, scan, psi, overwrite=False, **kwargs):
        return self.diffraction.adj_probe(
            psi=psi,
            scan=scan,
            nearplane=self.propagation.adj(
                farplane=farplane,
                overwrite=overwrite,
            ),
            overwrite=True,
        )

    def _compute_intensity(self, data, psi, scan, probe, n=-1, mode=None):
        """Compute detector intensities replacing the nth probe mode"""
        intensity = 0
        for m in range(probe.shape[-3]):
            intensity += np.sum(
                np.square(np.abs(self.fwd(
                    psi=psi,
                    scan=scan,
                    probe=mode if m == n else probe[..., m:m + 1, :, :],
                ).reshape(*data.shape[:2], -1, *data.shape[2:]))),
                axis=2,
            )  # yapf: disable
        return intensity

    def cost(self, data, psi, scan, probe, n=-1, mode=None):
        intensity = self._compute_intensity(data, psi, scan, probe, n, mode)
        return self.propagation.cost(data, intensity)

    def grad(self, data, psi, scan, probe):
        intensity = self._compute_intensity(data, psi, scan, probe)
        grad_obj = self.xp.zeros_like(psi)
        for mode in np.split(probe, probe.shape[-3], axis=-3):
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

    def grad_probe(self, data, psi, scan, probe, n=-1, mode=None):
        intensity = self._compute_intensity(data, psi, scan, probe, n, mode)
        return self.adj_probe(
            farplane=self.propagation.grad(
                data,
                self.fwd(
                    psi=psi,
                    scan=scan,
                    probe=mode if mode is not None else probe,
                ),
                intensity,
            ),
            psi=psi,
            scan=scan,
            overwrite=True,
        )

    # scatter dir to all GPUs
    def dir_multi(self, gpu_count, dir):  # lists of cupy array
        dir_cpu = self.asnumpy(dir)
        return self.asarray_multi(gpu_count, dir_cpu)

    # multi-GPU update()
    def update_multi(self, gpu_count, psi, gamma, dir):  # lists of cupy array
        for i in range(gpu_count):
            psi[i] = psi[i] + gamma * dir[i]
        return psi

    def grad_device(self, gpu_id, data, psi, scan, probe):  # cupy arrays
        return self.grad(data, psi, scan, probe)

    def cost_device(self, gpu_id, data, psi, scan, probe,
                    n=-1, mode=None):  # cupy arrays
        return self.cost(data, psi, scan, probe)

    # multi-GPU cost() entry point
    def cost_multi(self,
                   gpu_count,
                   data,
                   psi,
                   scan,
                   probe,
                   n=-1,
                   mode=None,
                   **kwargs):  # lists of cupy array
        psi_list = [None] * gpu_count
        for i in range(gpu_count):
            if 'step_length' in kwargs and 'dir' in kwargs:
                psi_list[i] = psi[i] + (kwargs.get('step_length') *
                                        kwargs.get('dir')[i])
            else:
                psi_list[i] = psi[i]

        gpu_list = range(gpu_count)
        with cf.ThreadPoolExecutor(max_workers=gpu_count) as executor:
            cost_out = executor.map(
                self.cost_device,
                gpu_list,
                data,
                psi_list,
                scan,
                probe,
            )
        cost_list = list(cost_out)

        cost_cpu = np.zeros(cost_list[0].shape, cost_list[0].dtype)
        for i in range(gpu_count):
            cost_cpu += Operator.asnumpy(cost_list[i])

        return cost_cpu

    # multi-GPU grad() entry point
    def grad_multi(self, gpu_count, data, psi, scan,
                   probe):  # lists of cupy array
        gpu_list = range(gpu_count)
        with cf.ThreadPoolExecutor(max_workers=gpu_count) as executor:
            grad_out = executor.map(
                self.grad_device,
                gpu_list,
                data,
                psi,
                scan,
                probe,
            )
        grad_list = list(grad_out)

        # grad_tmp = np.empty_like(grad_list[0])
        for i in range(1, gpu_count):
            # if cp.cuda.runtime.deviceCanAccessPeer(0, i):
            #     cp.cuda.runtime.deviceEnablePeerAccess(i)
            #     grad_tmp.data.copy_from_device(
            #         grad_list[i].data,
            #         grad_list[0].size * grad_list[0].itemsize,
            #     )
            # else:
            grad_cpu_tmp = self.asnumpy(grad_list[i])
            grad_tmp = self.asarray(grad_cpu_tmp)
            grad_list[0] += grad_tmp

        return grad_list[0]
