import numpy as np
from tike.opt import batch_indicies, randomizer
from tike.random import wobbly_center


def test_wobbly_center(N=20, k=5):
    """"Test that wobbly center generates better samples of the population.

    In this case 'better' is when the mean and std of the samples matches the
    population. So we compare the distance from the true mean/std to each
    sample mean/std.
    """
    m0 = [-np.sqrt(2), np.pi, np.e]
    s0 = [0.5, 3, 7]
    population = np.concatenate(
        [randomizer.normal(m, s, (N, 1)) for m, s in zip(m0, s0)],
        axis=1,
    )
    randomizer.shuffle(population, axis=0)

    def print_sample_error(samples):
        m = np.linalg.norm(
            [np.mean(population[x], axis=0) - m0 for x in samples])
        s = np.linalg.norm(
            [np.std(population[x], axis=0) - s0 for x in samples])
        print(
            f"distance from true mean: {m}\ndistance from true deviation: {s}")
        return m, s

    print('\nwobbly center')
    mw, sw = print_sample_error(wobbly_center(population, k))
    print('random sample')
    mr, sr, = print_sample_error(batch_indicies(N, k))

    assert mw < mr
    assert sw < sr
