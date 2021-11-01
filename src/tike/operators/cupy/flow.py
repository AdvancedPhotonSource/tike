__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

import cupy as cp
try:
    from importlib.resources import files
except ImportError:
    # Backport for python<3.9 available as importlib_resources package
    from importlib_resources import files

from .operator import Operator

_cu_source = files('tike.operators.cupy').joinpath('interp.cu').read_text()


def _remap_lanczos(Fe, x, m, F, fwd=True, cval=0.0):
    """Lanczos resampling from grid Fe to points x.

    At the edges, the Lanczos filter wraps around.

    Parameters
    ----------
    Fe : (H, W)
        The function at equally spaced samples.
    x : (N, 2) float32
        The non-uniform sample positions on the grid.
    m : int > 0
        The Lanczos filter is 2m + 1 wide.
    F : (N, )
        The values at the non-uniform samples.
    """
    assert Fe.ndim == 2
    assert x.ndim == 2 and x.shape[-1] == 2
    assert m > 0
    assert F.shape == x.shape[:-1], F.dtype == Fe.dtype
    assert Fe.dtype == 'complex64'
    assert F.dtype == 'complex64'
    assert x.dtype == 'float32'
    lanczos_width = 2 * m + 1

    if fwd:
        kernel = cp.RawKernel(_cu_source, "fwd_lanczos_interp2D")
    else:
        kernel = cp.RawKernel(_cu_source, "adj_lanczos_interp2D")

    grid = (-(-x.shape[0] // kernel.max_threads_per_block), 0, 0)
    block = (min(x.shape[0], kernel.max_threads_per_block), 0, 0)
    kernel(grid, block, (
        Fe,
        cp.array(Fe.shape, dtype='int32'),
        F,
        x,
        len(x),
        lanczos_width,
        cp.complex64(cval),
    ))


class Flow(Operator):
    """Map input 2D arrays to new coordinates by Lanczos interpolation.

    Uses Lanczos interpolation for a non-affine deformation of a series of 2D
    images.
    """

    def fwd(self, f, flow, filter_size=5, cval=0.0):
        """Remap individual pixels of f with Lanczos filtering.

        Parameters
        ----------
        f : (..., H, W) complex64
            A stack of arrays to be deformed.
        flow : (..., H, W, 2) float32
            The displacements to be applied to each pixel along the last two
            dimensions. Operation skipped when flow is None.
        filter_size : int
            The width of the Lanczos filter. Automatically rounded up to an
            odd positive integer.
        cval : complex64
            This value is used for interpolation from points outside the grid.
        """
        if flow is None:
            return f
        assert f.shape == flow.shape[:-1], (f.shape, flow.shape)
        # Convert from displacements to coordinates
        h, w = flow.shape[-3:-1]
        coords = -flow.copy()
        coords[..., 0] += self.xp.arange(h)[:, None]
        coords[..., 1] += self.xp.arange(w)

        # Reshape into stack of 2D images
        shape = f.shape
        coords = coords.reshape(-1, h * w, 2)
        f = f.reshape(-1, h, w)
        g = self.xp.zeros_like(f).reshape(-1, h * w)

        a = max(0, (filter_size) // 2)
        for i in range(len(f)):
            _remap_lanczos(f[i], coords[i], a, g[i], cval=cval)

        return g.reshape(shape)

    def adj(self, g, flow, filter_size=5, cval=0.0):
        """Remap individual pixels of f with Lanczos filtering.

        Parameters
        ----------
        g : (..., H, W) complex64
            A stack of deformed arrays.
        flow : (..., H, W, 2) float32
            The displacements to be applied to each pixel along the last two
            dimensions. Operation skipped when flow is None.
        filter_size : int
            The width of the Lanczos filter. Automatically rounded up to an
            odd positive integer.
        cval : complex64
            This value is used for interpolation from points outside the grid.
        """
        if flow is None:
            return g
        f = self.xp.zeros_like(g)
        assert f.shape == flow.shape[:-1], (f.shape, flow.shape)
        # Convert from displacements to coordinates
        h, w = flow.shape[-3:-1]
        coords = -flow.copy()
        coords[..., 0] += self.xp.arange(h)[:, None]
        coords[..., 1] += self.xp.arange(w)

        # Reshape into stack of 2D images
        shape = f.shape
        coords = coords.reshape(-1, h * w, 2)
        f = f.reshape(-1, h, w)
        g = g.reshape(-1, h * w)

        a = max(0, (filter_size) // 2)
        for i in range(len(f)):
            _remap_lanczos(f[i], coords[i], a, g[i], fwd=False, cval=cval)

        return f.reshape(shape)

    def inv(self, g, flow, filter_size=5, cval=0.0):
        return self.fwd(
            g,
            flow if flow is None else -flow,
            filter_size,
            cval,
        )
