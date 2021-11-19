import unittest
import numpy as np
import scipy.stats

from tike.opt import batch_indicies, randomizer
from tike.random import cluster_wobbly_center


class TestWobblyCenter(unittest.TestCase):

    def setUp(self, num_pop=500, num_cluster=10):
        """Generates a normally distributed 3D population."""
        self.num_pop = num_pop
        self.num_cluster = num_cluster
        m0 = [-np.sqrt(2), np.pi, np.e]
        s0 = [0.5, 3, 7]
        population = np.concatenate(
            [randomizer.normal(m, s, (num_pop, 1)) for m, s in zip(m0, s0)],
            axis=1,
        )
        randomizer.shuffle(population, axis=0)
        self.population = population

    def test_simple_cluster(self):
        references = [
            np.array([2, 3, 4, 9]),
            np.array([0, 5, 8]),
            np.array([1, 6, 7]),
        ]
        result = cluster_wobbly_center(np.arange(10)[:, None], 3)
        for a, b in zip(references, result):
            np.testing.assert_array_equal(a, b)

    def test_same_mean(self):
        """"Test that wobbly center generates better samples of the population.

        In this case 'better' is when the ANOVA test concludes that the samples
        are statistically the same on average. The ANOVA null hypothesis is
        that the sample are the same, so if p-values are > 0.05 we reject the
        alternative and keep the null hypothesis.
        """

        def print_sample_error(indices):
            """Apply ANOVA; test whether samples are statistically different."""
            F, p = scipy.stats.f_oneway(*[self.population[i] for i in indices])
            print(p)
            return p

        print('\nwobbly center')
        p0 = print_sample_error(
            cluster_wobbly_center(self.population, self.num_cluster))
        print('random sample')
        p1 = print_sample_error(batch_indicies(self.num_pop, self.num_cluster))

        # We should be more condifent that wobbly samples are the same
        assert np.all(p0 > p1)

    def test_no_clusters(self):
        with self.assertRaises(ValueError):
            cluster_wobbly_center(self.population, 0)
        with self.assertRaises(ValueError):
            cluster_wobbly_center(self.population, -1)

    def test_implementation_limited_clusters(self):
        with self.assertRaises(ValueError):
            cluster_wobbly_center(self.population, 0xFFFFFF)

    def test_one_cluster(self):
        samples = cluster_wobbly_center(self.population, 1)
        assert len(samples) == 1
        assert np.all(samples[0].flatten() - np.arange(self.num_pop) == 0)

    def test_more_clusters_than_population(self):
        with self.assertRaises(ValueError):
            cluster_wobbly_center(self.population, self.num_pop + 1)

    def test_max_clusters(self):
        samples = cluster_wobbly_center(self.population, self.num_pop)
        assert len(samples) == self.num_pop
        assert np.all(
            np.array(samples).flatten() - np.arange(self.num_pop) == 0)
