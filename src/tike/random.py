"""Provides random number generators for complex data types."""

import cupy as cp
import numpy as np


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
    NO_CLUSTER = 0xFFFF
    clusters = xp.full(len(population), NO_CLUSTER, dtype='uint16')
    clusters[starting_centroids] = range(num_cluster)
    unassigned = len(population) - len(starting_centroids)
    # print(f"\nStart with clusters: {clusters}")
    c = 0
    while True:
        # add the unclaimed observation that is furthest from this cluster
        if unassigned > 0:
            furthest = xp.argmax(
                xp.linalg.norm(
                    population[clusters == NO_CLUSTER] -
                    xp.mean(population[clusters == c], axis=0, keepdims=True),
                    axis=1,
                ),
                axis=0,
            )
            # i is the index of furthest in clusters
            i = xp.argmax(xp.cumsum(clusters == NO_CLUSTER) == (furthest + 1))
            # print(f"{i} will be added to {c}")
            unassigned -= 1
            clusters[i] = c
            # print(f"Start with clusters: {clusters}")
        else:
            return [xp.flatnonzero(clusters == c) for c in range(num_cluster)]
        c = (c + 1) % num_cluster
