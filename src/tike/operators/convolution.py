import itertools

from .operator import Operator


class Convolution(Operator):
    """2D Convolution operator with linear interpolation."""

    def __init__(self, probe_shape, nscan, nz, n, ntheta, **kwargs):
        super(Convolution, self).__init__(**kwargs)
        self.nscan = nscan
        self.probe_shape = probe_shape
        self.nz = nz
        self.n = n
        self.ntheta = ntheta

    def fwd(self, psi, scan):
        """Extract probe shaped patches from the psi at each scan position.

        The patches within the bounds of psi are linearly interpolated, and
        indices outside the bounds of psi are not allowed.
        """
        xp = self.array_module
        # For interpolating a pixel to a non-integer position on a grid, we need
        # to divide the area of the pixel between the 4 grid spaces that it
        # overlaps. The weights of each of the adjacent spaces is their area of
        # overlap with the pixel. The side lengths of each of the areas is the
        # remainder from the coordinates of the pixel on the grid.
        nearplane = xp.zeros(
            (self.ntheta, self.nscan, self.probe_shape, self.probe_shape),
            dtype=psi.dtype,
        )
        for view_angle in range(self.ntheta):
            for position in range(self.nscan):
                rem, ind = xp.modf(scan[view_angle, position])
                if (
                    ind[0] < 0 or ind[1] < 0
                    or psi.shape[-2] <= ind[0] + self.probe_shape
                    or psi.shape[-1] <= ind[1] + self.probe_shape
                ):
                    # skip scans where the probe position overlaps edges
                    continue
                w = (1 - rem[0], rem[0])
                l = (1 - rem[1], rem[1])
                areas = (w * l for w, l in itertools.product(w, l))
                x = (int(ind[0]), 1 + int(ind[0]))
                y = (int(ind[1]), 1 + int(ind[1]))
                corners = ((x, y) for x, y in itertools.product(x, y))
                for weight, (i, j) in zip(areas, corners):
                    nearplane[view_angle, position, ...] += psi[
                        view_angle,
                        i:i + self.probe_shape,
                        j:j + self.probe_shape,
                    ] * weight
        return nearplane

    def adj(self, nearplane, scan):
        """Combine probe shaped patches into a psi shaped grid by addition."""
        xp = self.array_module
        # See diffraction_fwd for description of interpolation algorithm.
        psi = xp.zeros_like(nearplane, shape=(self.ntheta, self.nz, self.n))
        # If nearplane cannot be reshaped into this shape, then there are not
        # enough scan positions to correctly do this operation.
        nearplane = nearplane.reshape(
            (self.ntheta, self.nscan, self.probe_shape, self.probe_shape),
        )
        for view_angle in range(self.ntheta):
            for position in range(self.nscan):
                rem, ind = xp.modf(scan[view_angle, position])
                if (
                    ind[0] < 0 or ind[1] < 0
                    or psi.shape[-2] <= ind[0] + self.probe_shape
                    or psi.shape[-1] <= ind[1] + self.probe_shape
                ):
                    continue
                w = (1 - rem[0], rem[0])
                l = (1 - rem[1], rem[1])
                areas = (w * l for w, l in itertools.product(w, l))
                x = (int(ind[0]), 1 + int(ind[0]))
                y = (int(ind[1]), 1 + int(ind[1]))
                corners = ((x, y) for x, y in itertools.product(x, y))
                for weight, (i, j) in zip(areas, corners):
                    psi[
                        view_angle,
                        i:i + self.probe_shape,
                        j:j + self.probe_shape,
                    ] += weight * nearplane[view_angle, position]
        return psi
