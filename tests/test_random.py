import numpy as np
import scipy.stats

from tike.opt import batch_indicies, randomizer
from tike.random import wobbly_center


def test_wobbly_center(N=500, k=10):
    """"Test that wobbly center generates better samples of the population.

    In this case 'better' is when the ANOVA test concludes that the samples are
    statistically the same on average. The ANOVA null hypothesis is that the
    sample are the same, so if p-values are > 0.05 we reject the alternative
    and keep the null hypothesis.
    """
    m0 = [-np.sqrt(2), np.pi, np.e]
    s0 = [0.5, 3, 7]
    population = np.concatenate(
        [randomizer.normal(m, s, (N, 1)) for m, s in zip(m0, s0)],
        axis=1,
    )
    randomizer.shuffle(population, axis=0)

    def print_sample_error(indices):
        """Apply ANOVA; test whether samples are statistically different."""
        F, p = scipy.stats.f_oneway(*[population[i] for i in indices])
        print(p)
        return p

    print('\nwobbly center')
    p0 = print_sample_error(wobbly_center(population, k))
    print('random sample')
    p1 = print_sample_error(batch_indicies(N, k))

    # We should be more condifent that wobbly samples are the same
    assert np.all(p0 > p1)
