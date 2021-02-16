import logging

import numpy as np

import tike.align
import tike.lamino
import tike.ptycho


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
