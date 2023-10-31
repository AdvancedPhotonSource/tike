import os
import unittest

import numpy as np
import scipy.stats

from tike.opt import batch_indicies
from tike.random import randomizer_np as randomizer
import tike.cluster


class ClusterTests():
    """Provides common tests for clustering methods."""

    cluster_method = staticmethod(None)

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

    def test_no_clusters(self):
        with self.assertRaises(ValueError):
            self.cluster_method(self.population, 0)
        with self.assertRaises(ValueError):
            self.cluster_method(self.population, -1)

    def test_implementation_limited_clusters(self):
        with self.assertRaises(ValueError):
            self.cluster_method(self.population, 0xFFFFFF)

    def test_one_cluster(self):
        samples = self.cluster_method(self.population, 1)
        assert len(samples) == 1
        assert np.all(samples[0].flatten() - np.arange(self.num_pop) == 0)

    def test_more_clusters_than_population(self):
        samples = self.cluster_method(self.population, self.num_pop + 1)
        assert len(samples) == self.num_pop + 1
        assert len(samples[-1]) == 0

    def test_max_clusters(self):
        samples = self.cluster_method(self.population, self.num_pop)
        assert len(samples) == self.num_pop
        assert np.all(
            np.array(samples).flatten() - np.arange(self.num_pop) == 0)

    def test_complete_set(self):
        samples = self.cluster_method(self.population, self.num_cluster)
        samples = np.sort(np.concatenate(samples))
        np.testing.assert_array_equal(np.arange(self.num_pop), samples)

    def test_some_clusters_are_singleton(self):
        samples = self.cluster_method(self.population, self.num_pop - 1)
        assert len(samples[0]) == 2
        assert len(samples[1]) == 1

    def test_clusters_sorted_by_size(self, num_cluster=7):
        samples = self.cluster_method(self.population, num_cluster)
        sizes = list(len(batch) for batch in samples)
        remainder = self.num_pop % num_cluster
        small = self.num_pop // num_cluster
        large = small + 1
        truth = [large] * remainder + [small] * (num_cluster - remainder)
        for a, b in zip(sizes, truth):
            assert a == b, f"Batch size {a} does not match expected size {b}."


class TestWobblyCenter(ClusterTests, unittest.TestCase):

    cluster_method = staticmethod(tike.cluster.wobbly_center)

    def test_simple_cluster(self):
        references = [
            np.array([2, 3, 4, 9]),
            np.array([0, 5, 8]),
            np.array([1, 6, 7]),
        ]
        result = tike.cluster.wobbly_center(np.arange(10)[:, None], 3)
        for a, b in zip(references, result):
            np.testing.assert_array_equal(a, b)

    def test_same_mean(self):
        """Test that wobbly center generates better samples of the population.

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
            tike.cluster.wobbly_center(self.population, self.num_cluster))
        print('random sample')
        np.random.seed(0)
        p1 = print_sample_error(batch_indicies(self.num_pop, self.num_cluster))

        # We should be more condifent that wobbly samples are the same
        assert np.all(p0 > p1)


class TestWobblyCenterRandomBootstrap(ClusterTests, unittest.TestCase):

    cluster_method = staticmethod(tike.cluster.wobbly_center_random_bootstrap)

    def test_simple_cluster(self):
        references = [
            np.array([2, 3, 4, 9]),
            np.array([0, 5, 8]),
            np.array([1, 6, 7]),
        ]
        result = tike.cluster.wobbly_center(np.arange(10)[:, None], 3)
        for a, b in zip(references, result):
            np.testing.assert_array_equal(a, b)

    def test_same_mean(self):
        """Test that wobbly center generates better samples of the population.

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
            tike.cluster.wobbly_center(self.population, self.num_cluster))
        print('random sample')
        np.random.seed(0)
        p1 = print_sample_error(batch_indicies(self.num_pop, self.num_cluster))

        # We should be more condifent that wobbly samples are the same
        assert np.all(p0 > p1)


class TestClusterCompact(ClusterTests, unittest.TestCase):

    cluster_method = staticmethod(tike.cluster.compact)

    def setUp(self, num_pop=50**2, num_cluster=10):
        """Generates points on a regular grid."""
        self.num_pop = num_pop
        self.num_cluster = num_cluster
        clusters = []
        population = np.stack(
            [x.flatten() for x in np.mgrid[0:100:2, 0:100:2]],
            axis=1,
        )
        randomizer.shuffle(population, axis=0)
        population = population[:num_pop]
        self.population = population

    def test_reduced_deviation(self):
        """Tests that compact clusters have smaller inter-cluster deviation."""

        def print_sample_error(indices):
            """Return the weighted generalized variance of the clusters.

            https://en.wikipedia.org/wiki/K-means_clustering#Description
            """
            cost = 0
            for c in indices:
                if len(c) > 1:
                    cost += abs(len(c) * np.linalg.det(
                        np.cov(
                            self.population[c],
                            rowvar=False,
                        )))
            print(cost)
            return cost

        print('\ncompact cluster')
        p0 = print_sample_error(
            tike.cluster.compact(self.population, self.num_cluster))
        print('random sample')
        np.random.seed(0)
        p1 = print_sample_error(batch_indicies(self.num_pop, self.num_cluster))

        # Every compact cluster should have smaller devation than a random
        # cluster
        assert p0 <= p1, f"assert {p0} <= {p1}"

    def test_plot_clusters(self):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        samples = tike.cluster.compact(self.population, self.num_cluster)
        plt.figure()
        for s in samples:
            plt.scatter(self.population[s][:, 0], self.population[s][:, 1])

        folder = os.path.join(os.path.dirname(__file__), 'result', 'random')
        if not os.path.isdir(folder):
            os.makedirs(folder)
        plt.savefig(os.path.join(folder, 'clusters.svg'))


class TestClusterStripesEqualCount(ClusterTests, unittest.TestCase):

    cluster_method = staticmethod(tike.cluster.stripes_equal_count)

    @unittest.skip("stripes_equal_count is not limited by the size of uint16.")
    def test_implementation_limited_clusters(self):
        pass

def test_split_by_scan():
    scan = np.mgrid[0:3, 0:3].reshape(2, -1)
    scan = np.moveaxis(scan, 0, -1)

    ind = tike.cluster.by_scan_stripes(scan, 3, axis=0)
    split = [scan[i] for i in ind]

    solution = [
        [[0, 0], [0, 1], [0, 2]],
        [[1, 0], [1, 1], [1, 2]],
        [[2, 0], [2, 1], [2, 2]],
    ]
    np.testing.assert_equal(split, solution)

    ind = tike.cluster.by_scan_stripes(scan, 3, axis=1)
    split = [scan[i] for i in ind]
    solution = [
        [[0, 0], [1, 0], [2, 0]],
        [[0, 1], [1, 1], [2, 1]],
        [[0, 2], [1, 2], [2, 2]],
    ]
    np.testing.assert_equal(split, solution)
