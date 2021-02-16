import logging

import numpy as np

import tike.align
import tike.lamino
import tike.ptycho


def update_penalty(comm, psi, h, h0, rho, diff=10):
    r = np.linalg.norm(psi - h)**2
    s = np.linalg.norm(rho * (h - h0))**2
    r, s = [comm.gather(x) for x in ([r], [s])]
    if comm.rank == 0:
        r = np.sum(r)
        s = np.sum(s)
        if (r > diff * s):
            rho *= 2
        elif (s > diff * r):
            rho *= 0.5
    rho = comm.broadcast(rho)
    logging.info(f"Update penalty parameter ρ = {rho}.")
    return rho


def find_min_max(data):
    mmin = np.zeros(data.shape[0], dtype='float32')
    mmax = np.zeros(data.shape[0], dtype='float32')

    for k in range(data.shape[0]):
        h, e = np.histogram(data[k][:], 1000)
        stend = np.where(h > np.max(h) * 0.005)
        st = stend[0][0]
        end = stend[0][-1]
        mmin[k] = e[st]
        mmax[k] = e[end + 1]

    return mmin, mmax


def simulate(
    u,
    scan,
    probe,
    flow,
    angle,
    tilt,
    theta,
    padded_shape,
    detector_shape,
):
    phi = np.exp(1j * tike.lamino.simulate(
        obj=u,
        tilt=tilt,
        theta=theta,
    ))
    psi = tike.align.simulate(
        original=phi,
        flow=flow,
        padded_shape=padded_shape,
        angle=angle,
        cval=1.0,
    )
    data = tike.ptycho.simulate(
        psi=psi,
        probe=probe,
        detector_shape=detector_shape,
        scan=scan,
    )
    return data, psi, phi


def print_log_line(**kwargs):
    """Print keyword arguments and values on a single comma-separated line.

    The format of the line is as follows:

    ```
    foo: 003, bar: +1.234e+02, hello: world\n
    ```

    Parameters
    ----------
    line: dictionary
        The key value pairs to be printed.

    """
    line = []
    for k, v in kwargs.items():
        # Use special formatting for float and integers
        if isinstance(v, (float, np.floating)):
            line.append(f'"{k}": {v:6.3e}')
        elif isinstance(v, (int, np.integer)):
            line.append(f'"{k}": {v:3d}')
        else:
            line.append(f'"{k}": {v}')
    # Combine all the strings and strip the last comma
    print("{", ", ".join(line), "}", flush=True)


def optical_flow_tvl1(unaligned, original, num_iter=16):
    """Wrap scikit-image optical_flow_tvl1 for complex values"""
    from skimage.registration import optical_flow_tvl1
    iflow = [
        optical_flow_tvl1(
            original[i].imag,
            unaligned[i].imag,
            num_iter=num_iter,
        ) for i in range(len(original))
    ]
    rflow = [
        optical_flow_tvl1(
            original[i].real,
            unaligned[i].real,
            num_iter=num_iter,
        ) for i in range(len(original))
    ]
    flow = np.array(rflow, dtype='float32') + np.array(iflow, dtype='float32')
    flow = np.moveaxis(flow, 1, -1) / 2.0
    return flow


def center_of_mass(m, axis=None):
    """Return the center of mass of m along the given axis.

    Parameters
    ----------
    m : array
        Values to find the center of mass from
    axis : tuple(int)
        The axes to find center of mass along.

    Returns
    -------
    center : (..., len(axis)) array[int]
        The shape of center is the shape of m with the dimensions corresponding
        to axis removed plus a new dimension appended whose length is the
        of length of axis in the order of axis.

    """
    centers = []
    for a in range(m.ndim) if axis is None else axis:
        shape = np.ones_like(m.shape)
        shape[a] = m.shape[a]
        x = np.arange(1, m.shape[a] + 1).reshape(*shape).astype(m.dtype)
        centers.append((m * x).sum(axis=axis) / m.sum(axis=axis) - 1)
    return np.stack(centers, axis=-1)
