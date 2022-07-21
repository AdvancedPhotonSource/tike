import optparse
import numpy as np
import tike.ptycho
from tike.ptycho.position import PositionOptions


def test_position_join(N=245, num_batch=11):

    scan = np.random.rand(N, 2)
    assert scan.shape == (N, 2)
    indices = np.arange(N)
    assert np.amin(indices) == 0
    assert np.amax(indices) == N - 1
    np.random.shuffle(indices)
    batches = np.array_split(indices, num_batch)

    opts = tike.ptycho.PositionOptions(
        scan,
        use_adaptive_moment=True,
    )

    optsb = [opts.split(b) for b in batches]

    # Copies non-array params into new object
    new_opts = optsb[0].split([])

    for b, i in zip(optsb, batches):
        new_opts = new_opts.join(b, i)

    np.testing.assert_array_equal(
        new_opts.initial_scan,
        opts.initial_scan,
    )

    np.testing.assert_array_equal(
        new_opts._momentum,
        opts._momentum,
    )
