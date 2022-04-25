"""Provides random number generators for complex data types."""

import logging

import cupy as cp
import numpy as np


logger = logging.getLogger(__name__)


def numpy_complex(*shape):
    """Return a complex random array in the range [-0.5, 0.5)."""
    return (np.random.rand(*shape, 2) - 0.5).view('complex')[..., 0]


def cupy_complex(*shape):
    """Return a complex random array in the range [-0.5, 0.5)."""
    return (cp.random.rand(*shape, 2) - 0.5).view('complex')[..., 0]


def cluster_wobbly_center(population, num_cluster):
    """Return the indices that divide population into heterogenous clusters.

    Uses a contrarian approach to clustering by maximizing the heterogeneity
    inside each cluster to ensure that each cluster would be able to capture
    the entire variance of the original population yielding clusters which are
    similar to each other in excess to the original population itself.

    Parameters
    ----------
    population : (M, N) array_like
        The M samples of an N dimensional population that needs to be clustered.
    num_cluster : int (0..M]
        The number of clusters in which to divide M samples.

    Returns
    -------
    indicies : (num_cluster,) list of array of integer
        The indicies of population that belong to each cluster.

    Raises
    ------
    ValueError
        If num_cluster is less than 1 or more than 65535. The implementation
        uses uint16 as cluster tag, so it cannot count more than that number of
        clusters.

    References
    ----------
    Mishra, Megha, Chandrasekaran Anirudh Bhardwaj, and Kalyani Desikan. "A
    Maximal Heterogeneity Based Clustering Approach for Obtaining Samples."
    arXiv preprint arXiv:1709.01423 (2017).
    """
    logger.info("Clustering method is wobbly center.")
    xp = cp.get_array_module(population)
    if num_cluster == 1 or num_cluster == population.shape[0]:
        return xp.split(xp.arange(population.shape[0]), num_cluster)
    if not 0 < num_cluster <= min(0xFFFF, population.shape[0]):
        raise ValueError(
            f"The number of clusters must be 0 < {num_cluster} < min(65536, M)."
        )
    # Start with the num_cluster observations closest to the global centroid
    starting_centroids = xp.argpartition(
        xp.linalg.norm(population - xp.mean(population, axis=0, keepdims=True),
                       axis=1),
        num_cluster,
        axis=0,
    )[:num_cluster]
    # Use a label array to keep track of cluster assignment
    UNASSIGNED = 0xFFFF
    labels = xp.full(len(population), UNASSIGNED, dtype='uint16')
    labels[starting_centroids] = range(num_cluster)
    # print(f"\nStart with labels: {labels}")
    for c in range(len(population) - len(starting_centroids)):
        # c is the label of the cluster getting the next addition
        c = c % num_cluster
        # add the unclaimed observation that is furthest from this cluster
        furthest = xp.argmax(
            xp.linalg.norm(
                population[labels == UNASSIGNED] -
                xp.mean(population[labels == c], axis=0, keepdims=True),
                axis=1,
            ),
            axis=0,
        )
        # i is the index of furthest in labels
        i = xp.argmax(xp.cumsum(labels == UNASSIGNED) == (furthest + 1))
        # print(f"{i} will be added to {c}")
        labels[i] = c
        # print(f"Start with labels: {labels}")
    return [xp.flatnonzero(labels == c) for c in range(num_cluster)]


def cluster_compact(population, num_cluster, max_iter=500):
    """Return the indices that divide population into compact clusters.

    Uses an approach that is inspired by the naive k-means algorithm, but it
    returns equally sized clusters.

    Parameters
    ----------
    population : (M, N) array_like
        The M samples of an N dimensional population that needs to be clustered.
    num_cluster : int (0..M]
        The number of clusters in which to divide M samples.

    Returns
    -------
    indicies : (num_cluster,) list of array of integer
        The indicies of population that belong to each cluster.

    Raises
    ------
    ValueError
        If num_cluster is less than 1 or more than 65535. The implementation
        uses uint16 as cluster tag, so it cannot count more than that number of
        clusters.
    """
    logger.info("Clustering method is compact.")
    # Indexing and serial operations is very slow on GPU, so always use host
    population = cp.asnumpy(population)
    if num_cluster == 1 or num_cluster == population.shape[0]:
        return np.split(np.arange(population.shape[0]), num_cluster)
    if not 0 < num_cluster <= min(0xFFFF, population.shape[0]):
        raise ValueError(
            f"The number of clusters must be 0 < {num_cluster} < min(65536, M)."
        )
    # Define a constant array used for indexing.
    _all = np.arange(len(population))
    # Specify the number of points allowed in each cluster
    size = np.zeros(num_cluster, dtype='int')
    max_size = np.full(num_cluster, len(population) // num_cluster)
    max_size[:len(population) % num_cluster] += 1

    # Use kmeans++ to choose initial cluster centers
    starting_centroids = np.zeros(num_cluster, dtype='int')
    starting_centroids[0] = np.random.choice(_all, size=1, p=None)[0]
    distances = np.inf
    for c in range(1, num_cluster):
        distances = np.minimum(
            distances,
            np.linalg.norm(
                population - population[starting_centroids[c - 1]],
                axis=1,
            )**2,
        )
        starting_centroids[c] = np.random.choice(
            _all,
            size=1,
            p=distances / distances.sum(),
        )[0]
    centroids = population[starting_centroids]

    # Use a label array to keep track of cluster assignment
    UNASSIGNED = 0xFFFF
    labels = np.full(len(population), UNASSIGNED, dtype='uint16')

    # Add all points to initial clusters
    distances = np.empty((len(population), num_cluster))
    unfilled_clusters = list(range(num_cluster))
    _unassigned = list(range(len(population)))
    for c in unfilled_clusters:
        distances[:, c] = np.linalg.norm(centroids[c] - population, axis=1)
        p = starting_centroids[c]
        labels[p] = c
        _unassigned.remove(p)
        size[c] += 1
    while unfilled_clusters:
        nearest = np.array(unfilled_clusters)[np.argmin(
            distances[:, unfilled_clusters],
            axis=1,
        )]
        farthest = np.array(unfilled_clusters)[np.argmax(
            distances[:, unfilled_clusters],
            axis=1,
        )]
        priority = np.array(_unassigned)[np.argsort(
            (distances[_all, nearest] -
             distances[_all, farthest])[_unassigned])]
        for p in priority:
            assert labels[p] == UNASSIGNED
            labels[p] = nearest[p]
            _unassigned.remove(p)
            assert size[nearest[p]] < max_size[nearest[p]]
            size[nearest[p]] += 1
            if size[nearest[p]] == max_size[nearest[p]]:
                unfilled_clusters.remove(nearest[p])
                # re-start with one less available cluster
                break

    if __debug__:
        old_objective = _k_means_objective(population, labels, num_cluster)

    # Swap points between clusters to minimize objective
    for _ in range(max_iter):
        any_were_swapped = False
        # Determine distance from each point to each cluster
        for c in range(num_cluster):
            distances[:, c] = np.linalg.norm(centroids[c] - population, axis=1)
        # Determine if each point would be happier in another cluster.
        # Negative happiness is bad; zero is optimal.
        labels_wanted = np.argmin(distances, axis=1)
        happiness = distances[_all, labels_wanted] - distances[_all, labels]
        # Starting from the least happy point, swap points between groups to
        # improve net happiness
        for p in np.argsort(happiness):
            if happiness[p] < 0:
                # Compute the change in happiness from swapping this point with
                # every other point
                net_happiness = (
                    + distances[p, labels[p]]
                    + distances[_all, labels]
                    - distances[p, labels]
                    - distances[_all, labels[p]]
                )  # yapf: disable
                good_swaps = np.flatnonzero(
                    np.logical_and(
                        # only want swaps that improve happiness
                        net_happiness > 0,
                        # swapping within a cluster has no effect
                        labels != labels[p],
                    ))
                if good_swaps.size > 0:
                    any_were_swapped = True
                    o = good_swaps[np.argmax(net_happiness[good_swaps])]
                    assert labels[p] != labels[
                        o], 'swapping within a cluster has no effect!'
                    labels[o], labels[p] = labels[p], labels[o]
                    happiness[o] = distances[o, labels_wanted[o]] - distances[
                        o, labels[o]]
                    happiness[p] = distances[p, labels_wanted[p]] - distances[
                        p, labels[p]]
        if not any_were_swapped:
            break
        elif __debug__:
            objective = _k_means_objective(population, labels, num_cluster)
            # print(f"{objective:e}")
            # NOTE: Assertion disabled because happiness metric is heuristic
            # approximation of k-means objective
            # assert old_objective >= objective, (old_objective, objective)
            old_objective = objective

        # compute new cluster centroids
        for c in range(num_cluster):
            centroids[c] = np.mean(population[labels == c], axis=0)

    if __debug__:
        for c in range(num_cluster):
            _assert_cluster_is_full(labels, c, max_size[c])
    return [np.flatnonzero(labels == c) for c in range(num_cluster)]


def _k_means_objective(population, labels, num_cluster):
    """Return the weighted sum of the generalized variance of each cluster."""
    xp = cp.get_array_module(population)
    cost = 0
    for c in range(num_cluster):
        cost += xp.sum(labels == c) * xp.linalg.det(
            xp.cov(
                population[labels == c],
                rowvar=False,
            ))
    return cost


def _assert_cluster_is_full(labels, c, size):
    xp = cp.get_array_module(labels)
    assert size == np.sum(labels == c), ('All clusters should be full, but '
                                         f'cluster {c} had '
                                         f'{xp.sum(labels == c)} points '
                                         f'when it should have {size}.')
