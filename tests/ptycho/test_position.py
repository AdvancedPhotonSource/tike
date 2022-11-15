import numpy as np
import tike.ptycho
import tike.linalg


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


def test_fit_affine(N=213):

    truth = np.random.rand(4).tolist()
    T = tike.ptycho.AffineTransform(*truth)
    error = np.random.normal(size=(N, 2))
    positions0 = (np.random.rand(*(N, 2)) - 0.5) * 1000
    positions1 = positions0 @ T.asarray() + error
    weights = 1 / (1 + np.square(error).sum(axis=-1))

    result = tike.ptycho.estimate_global_transformation(
        positions0,
        positions1,
        weights,
    )

    np.testing.assert_allclose(
        result.astuple(),
        truth,
        rtol=5e-2,
    )
