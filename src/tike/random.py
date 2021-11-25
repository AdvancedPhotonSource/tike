"""Provides random number generators for complex data types."""

import cupy as cp
import numpy as np
from numpy.random.mtrand import random

from tike.opt import randomizer


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


def cluster_compact(population, num_cluster):
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
    xp = cp.get_array_module(population)
    if num_cluster == 1 or num_cluster == population.shape[0]:
        return xp.split(xp.arange(population.shape[0]), num_cluster)
    if not 0 < num_cluster <= min(0xFFFF, population.shape[0]):
        raise ValueError(
            f"The number of clusters must be 0 < {num_cluster} < min(65536, M)."
        )
    # Specify the number of points allowed in each cluster
    max_size = xp.full(num_cluster, len(population) // num_cluster)
    max_size[:len(population) % num_cluster] += 1
    # Start with random num_cluster observations as centroid
    starting_centroids = randomizer.permutation(len(population))[:num_cluster]
    centroids = population[starting_centroids]
    # Use a label array to keep track of cluster assignment
    # Must start with every point assigned to a cluster
    labels = xp.arange(len(population), dtype='uint16') % num_cluster
    for c in range(num_cluster):
        _assert_cluster_is_full(labels, c, max_size[c])
    labels_old = None
    # Define a constant array used for indexing.
    _all = xp.arange(len(population))
    while xp.any(labels != labels_old):
        labels_old = labels.copy()
        # Determine distance from each point to each cluster
        distances = xp.empty((len(population), num_cluster))
        for c in range(num_cluster):
            distances[:, c] = xp.linalg.norm(centroids[c] - population, axis=1)
        labels_wanted = xp.argmin(distances, axis=1)
        # Determine if each point would be happier in another cluster.
        # Negative happiness is bad; zero is optimal.
        happiness = distances[_all, labels_wanted] - distances[_all, labels]
        # Starting from the least happy, move points to new groups
        for p in np.argsort(happiness):
            _assert_cluster_is_full(
                labels,
                labels_wanted[p],
                max_size[labels_wanted[p]],
            )
            # Search the wanted cluster for another point that labels_wanted to swap
            others = xp.flatnonzero(
                xp.logical_and(
                    labels == labels_wanted[p],
                    labels_wanted == labels[p],
                ))
            if len(others) > 0:
                labels[others[0]], labels[p] = labels_wanted[
                    others[0]], labels_wanted[p]
                _assert_cluster_is_full(
                    labels,
                    labels_wanted[p],
                    max_size[labels_wanted[p]],
                )
        # compute new cluster centroids
        for c in range(num_cluster):
            centroids[c] = xp.mean(population[labels == c], axis=0)

    return [xp.flatnonzero(labels == c) for c in range(num_cluster)]


def _assert_cluster_is_full(labels, c, size):
    assert size == np.sum(labels == c), ('All clusters should be full, but '
                                         f'cluster {c} had '
                                         f'{np.sum(labels == c)} points '
                                         f'when it should have {size}.')
