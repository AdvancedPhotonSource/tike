"""Provides random number generators for complex data types."""

import cupy as cp
import numpy as np


def numpy_complex(*shape):
    """Return a complex random array in the range [-0.5, 0.5)."""
    return (np.random.rand(*shape, 2) - 0.5).view('complex')[..., 0]


def cupy_complex(*shape):
    """Return a complex random array in the range [-0.5, 0.5)."""
    return (cp.random.rand(*shape, 2) - 0.5).view('complex')[..., 0]


def wobbly_center(obs, k, xp=cp):
    """Return k clusters with maximal dissimilarity inside each cluster.

    Uses a contrarian approach to clustering by maximizing the heterogeneity
    inside each cluster to ensure that each cluster would be able to capture
    the entire variance of the original population yielding clusters which are
    similar to each other in excess to the original population itself.

    Parameters
    ----------
    obs : (M, N) array_like
        The N dimensional population of N samples that needs to be clustered.
    k : int < M
        The number of clusters in which to divide M.

    Returns
    -------
    indicies : (k,) list of array of boolean
        The indicies of obs that belong to each cluster.

    References
    ----------
    Mishra, Megha, Chandrasekaran Anirudh Bhardwaj, and Kalyani Desikan. "A
    Maximal Heterogeneity Based Clustering Approach for Obtaining Samples."
    arXiv preprint arXiv:1709.01423 (2017).
    """
    xp = cp.get_array_module(obs)
    # Start with the k observations closest to the global centroid
    starting_centroids = xp.argpartition(
        xp.linalg.norm(obs - xp.mean(obs, axis=0, keepdims=True), axis=1),
        k,
        axis=0,
    )[:k]
    # Use a label array to keep track of cluster assignment; 255 is no cluster
    clusters = xp.empty(len(obs), dtype='uint8')
    clusters[:] = 255
    clusters[starting_centroids] = range(k)
    unassigned = len(obs) - len(starting_centroids)
    # print(f"\nStart with clusters: {clusters}")
    c = 0
    while True:
        # add the unclaimed observation that is furthest from this cluster
        if unassigned > 0:
            furthest = xp.argmax(
                xp.linalg.norm(
                    obs[clusters == 255] -
                    xp.mean(obs[clusters == c], axis=0, keepdims=True),
                    axis=1,
                ),
                axis=0,
            )
            l = xp.argmax(xp.cumsum(clusters == 255) == (furthest + 1))
            # print(f"{l} will be added to {c}")
            unassigned -= 1
            clusters[l] = c
            # print(f"Start with clusters: {clusters}")
        else:
            return [xp.flatnonzero(clusters == c) for c in range(k)]
        c = (c + 1) % k
