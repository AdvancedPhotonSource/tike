import itertools
import typing
import logging

import cupy as cp
import cupyx
import numpy as np
import numpy.typing as npt

import tike.communicators

logger = logging.getLogger(__name__)


def _split_gpu(
    m: npt.NDArray,
    x: npt.ArrayLike,
    dtype: npt.DTypeLike,
) -> npt.ArrayLike:
    return cp.asarray(x[m], dtype=dtype)


def _split_host(
    m: npt.NDArray,
    x: npt.ArrayLike,
    dtype: npt.DTypeLike,
) -> npt.ArrayLike:
    return np.asarray(x[m], dtype=dtype)


def _split_pinned(
    m: npt.NDArray,
    x: npt.ArrayLike,
    dtype: npt.DTypeLike,
) -> npt.ArrayLike:
    pinned = cupyx.empty_pinned(shape=(len(m), *x.shape[1:]), dtype=dtype)
    pinned[...] = x[m]
    return pinned


def by_scan_grid(
    *args,
    pool: tike.communicators.ThreadPool,
    shape: typing.Tuple[int],
    dtype: typing.List[npt.DTypeLike],
    destination: typing.List[str],
    scan: npt.NDArray[np.float32],
    fly: int = 1,
):
    """Split the field of view into a 2D grid.

    Mask divide the data into a 2D grid of spatially contiguous regions.

    Parameters
    ----------
    shape : tuple of int
        The number of grid divisions along each dimension.
    dtype : List[str]
        The datatypes of the args after splitting.
    scan : (nscan, 2) float32
        The 2D coordinates of the scan positions.
    args : (nscan, ...) float32 or None
        The arrays to be split by scan position.
    fly : int
        The number of scan positions per frame.

    Returns
    -------
    order : List[array[int]]
        The locations of the inputs in the original arrays.
    scan : List[array[float32]]
        The divided 2D coordinates of the scan positions.
    args : List[array[float32]] or None
        Each input divided into regions or None if arg was None.

    """
    if len(shape) != 2:
        raise ValueError('The grid shape must have two dimensions.')
    vstripes = by_scan_stripes(scan, shape[0], axis=0, fly=fly)
    hstripes = by_scan_stripes(scan, shape[1], axis=1, fly=fly)
    mask = [
        np.logical_and(*pair) for pair in itertools.product(vstripes, hstripes)
    ]

    order = np.arange(scan.shape[-2])
    order = [order[m] for m in mask]

    split_args = []
    for arg, t, dest in zip([scan, *args], dtype, destination):
        if arg is None:
            split_args.append(None)
        else:
            split_args.append(
                pool.map(
                    _split_gpu if dest == 'gpu' else _split_pinned,
                    mask,
                    x=arg,
                    dtype=t,
                ))

    return (order, *split_args)


def by_scan_stripes(
    scan,
    n: int,
    fly: int = 1,
    axis: int = 0,
) -> typing.List[npt.NDArray[np.bool_]]:
    """Return `n` boolean masks that split the field of view into stripes.

    Mask divide the data into spatially contiguous regions along the position
    axis.

    Split scan into three stripes:
    >>> [scan[s] for s in by_scan_stripes(scan, 3)]

    FIXME: Only uses the first view to divide the positions. Assumes the
    positions on all angles are distributed similarly.

    Parameters
    ----------
    scan : (nscan, 2) float32
        The 2D coordinates of the scan positions.
    n : int
        The number of stripes.
    fly : int
        The number of scan positions per frame.
    axis : int (0 or 1)
        Which spatial dimension to divide along. i.e. horizontal or vertical.

    Returns
    -------
    mask : list of (nscan, ) boolean
        A list of boolean arrays which divide the scan positions into `n`
        stripes.

    """
    if scan.ndim != 2:
        raise ValueError('scan must have two dimensions.')
    if n < 1:
        raise ValueError('The number of stripes must be > 0.')

    nscan, _ = scan.shape
    if (nscan // fly) * fly != nscan:
        raise ValueError('The number of scan positions must be an '
                         'integer multiple of the number of fly positions.')

    # Reshape scan so positions in the same fly scan are not separated
    scan = scan.reshape(nscan // fly, fly, 2)

    # Determine the edges of the horizontal stripes
    edges = np.linspace(
        scan[..., axis].min(),
        scan[..., axis].max(),
        n + 1,
        endpoint=True,
    )

    # Move the outer edges to include all points
    edges[0] -= 1
    edges[-1] += 1

    # Generate masks which put points into stripes
    return [
        np.logical_and(
            edges[i] < scan[:, 0, axis],
            scan[:, 0, axis] <= edges[i + 1],
        ).repeat(fly) for i in range(n)
    ]


def by_scan_stripes_contiguous(
    *args,
    pool: tike.communicators.ThreadPool,
    shape: typing.Tuple[int],
    dtype: typing.List[npt.DTypeLike],
    destination: typing.List[str],
    scan: npt.NDArray[np.float32],
    fly: int = 1,
    batch_method,
    num_batch: int,
) -> typing.Tuple[typing.List[npt.NDArray],
                  typing.List[typing.List[npt.NDArray]]]:
    """Split data by into stripes and create contiguously ordered batches.

    Divide the field of view into one stripe per devices; within each stripe,
    create batches according to the batch_method loading the batches into
    contiguous blocks in device memory.

    Parameters
    ----------
    shape : tuple of int
        The number of grid divisions along each dimension.
    dtype : List[str]
        The datatypes of the args after splitting.
    scan : (nscan, 2) float32
        The 2D coordinates of the scan positions.
    args : (nscan, ...) float32 or None
        The arrays to be split by scan position.
    fly : int
        The number of scan positions per frame.
    batch_method :
        The method for determining the batches after dividing amongst GPUs

    Returns
    -------
    order : List[array[int]]
        The locations of the inputs in the original arrays.
    batches : List[List[array[int]]]
        The locations of the elements of each batch
    scan : List[array[float32]]
        The divided 2D coordinates of the scan positions.
    args : List[array[float32]] or None
        Each input divided into regions or None if arg was None.

    """
    if len(shape) != 2:
        raise ValueError('The grid shape must have two dimensions.')

    map_to_gpu = stripes_equal_count(
        population=scan,
        num_cluster=shape[0] * shape[1],
        dim=0,
    )
    split_scan = pool.map(
        _split_host,
        map_to_gpu,
        x=scan,
        dtype=scan.dtype,
    )
    batches_noncontiguous: typing.List[typing.List[npt.NDArray]] = pool.map(
        getattr(tike.cluster, batch_method),
        split_scan,
        num_cluster=num_batch,
    )
    map_to_gpu_contiguous: typing.List[npt.NDArray] = []
    batches_contiguous: typing.List[typing.List[npt.NDArray]] = []
    for gpu_map, batch_map in zip(map_to_gpu, batches_noncontiguous):
        batch_indices = gpu_map[np.concatenate(batch_map)]
        map_to_gpu_contiguous.append(batch_indices)
        batch_sizes = [len(batch) for batch in batch_map]
        batch_breaks = np.cumsum(batch_sizes)[:-1]
        batches_contiguous.append(
            np.array_split(
                np.arange(len(batch_indices)),
                batch_breaks,
            ))

    split_args = []
    for arg, t, dest in zip([scan, *args], dtype, destination):
        if arg is None:
            split_args.append(None)
        else:
            split_args.append(
                pool.map(
                    _split_gpu if dest == 'gpu' else _split_pinned,
                    map_to_gpu_contiguous,
                    x=arg,
                    dtype=t,
                ))

    if __debug__:
        for device in batches_contiguous:
            assert len(device) == num_batch, (
                f"There should be {num_batch} batches, found {len(device)}"
            )

    return (map_to_gpu_contiguous, batches_contiguous, *split_args)


def stripes_equal_count(
    population: npt.ArrayLike,
    num_cluster: int,
    dim: int = 0,
) -> typing.List[npt.NDArray]:
    """Return indices dividing the population into stripes of equal count.

    The returned clusters are divided along the provided dimension into
    clusters of approximate equal numbers of elements.

    Parameters
    ----------
    population : (M, N) array_like
        The M samples of an N dimensional population that needs to be
        clustered.
    num_cluster : int (0..M]
        The number of clusters in which to divide M samples.
    dim : int
        The dimension (of N) along which the population is divided.

    Returns
    -------
    indicies : (num_cluster,) list of array of integer
        The indicies of population that belong to each cluster.
    """
    logger.info("Clustering method is stripes.")
    xp = cp.get_array_module(population)
    if (num_cluster == 1) or (num_cluster >= len(population)):
        return np.array_split(np.arange(population.shape[0]), num_cluster)
    # Sort the population along the dimension, then split into ranges of approx
    # equal size
    return np.array_split(
        cp.asnumpy(xp.argsort(population[:, dim])),
        num_cluster,
    )


def wobbly_center(population, num_cluster):
    """Return the indices that divide population into heterogenous clusters.

    Uses a contrarian approach to clustering by maximizing the heterogeneity
    inside each cluster to ensure that each cluster would be able to capture
    the entire variance of the original population yielding clusters which are
    similar to each other in excess to the original population itself.

    Parameters
    ----------
    population : (M, N) array_like
        The M samples of an N dimensional population that needs to be
        clustered.
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
    if not 0 < num_cluster < 0xFFFF:
        raise ValueError(
            f"The number of clusters must be 0 < {num_cluster} < 65536."
        )
    if (num_cluster == 1) or (num_cluster >= len(population)):
        return np.array_split(np.arange(population.shape[0]), num_cluster)
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
    return [cp.asnumpy(xp.flatnonzero(labels == c)) for c in range(num_cluster)]


def wobbly_center_random_bootstrap(
    population,
    num_cluster: int,
    boot_fraction: float = 0.95,
) -> typing.List[npt.NDArray]:
    """Return the indices that divide population into heterogenous clusters.

    Uses a hybrid approach to generate heterogenous clusters. First, a fraction
    of the population is divided into clusters randomly, then the wobbly center
    algorithm is used to distriube the remaining segment of the population with
    the goal of maximizing intracluster heterogeneity.

    Parameters
    ----------
    population : (M, N) array_like
        The M samples of an N dimensional population that needs to be
        clustered.
    num_cluster : int (0..M]
        The number of clusters in which to divide M samples.
    boot_fraction: (0, 1]
        The percentage of each cluster that is randomly assigned before
        starting the wobbly center algorithm.

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
    logger.info("Clustering method is wobbly center with random bootstrap.")
    xp = cp.get_array_module(population)
    if not 0 < num_cluster < 0xFFFF:
        raise ValueError(
            f"The number of clusters must be 0 < {num_cluster} < 65536."
        )
    if (num_cluster == 1) or (num_cluster >= len(population)):
        return np.array_split(np.arange(population.shape[0]), num_cluster)
    # Partially initialize the clusters randomly; each cluster starts with an
    # equal number of members
    num_bootstrap = int(len(population) * boot_fraction)
    num_bootstrap -= num_bootstrap % num_cluster
    seed = xp.random.choice(
        len(population),
        size=num_bootstrap,
        replace=False,
    )
    # Use a label array to keep track of cluster assignment
    UNASSIGNED = 0xFFFF
    labels = xp.full(len(population), UNASSIGNED, dtype='uint16')
    for c in range(num_cluster):
        labels[seed[c::num_cluster]] = c
    # print(f"\nStart with labels: {labels}")
    for c in range(len(population) - num_bootstrap):
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
    return [cp.asnumpy(xp.flatnonzero(labels == c)) for c in range(num_cluster)]


def compact(population, num_cluster, max_iter=500):
    """Return the indices that divide population into compact clusters.

    Uses an approach that is inspired by the naive k-means algorithm, but it
    returns equally sized clusters.

    Parameters
    ----------
    population : (M, N) array_like
        The M samples of an N dimensional population that needs to be
        clustered.
    num_cluster : int (0..M]
        The number of clusters in which to divide M samples.

    Returns
    -------
    indicies : (num_cluster,) list of array of integer
        The indicies of population that belong to each cluster. Clusters are
        sorted from largest to smallest.

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
    if not 0 < num_cluster < 0xFFFF:
        raise ValueError(
            f"The number of clusters must be 0 < {num_cluster} < 65536."
        )
    if (num_cluster == 1) or (num_cluster >= len(population)):
        return np.array_split(np.arange(population.shape[0]), num_cluster)
    # Define a constant array used for indexing.
    _all = np.arange(len(population))
    # Specify the number of points allowed in each cluster
    size = np.zeros(num_cluster, dtype='int')
    max_size = np.full(num_cluster, len(population) // num_cluster)
    max_size[:len(population) % num_cluster] += 1
    assert np.sum(max_size) == len(population), (
        f"Sum of cluster maximums {np.sum(max_size)} should "
        f"equal the population size {len(population)}!")

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
    for c in range(num_cluster):
        if size[c] >= max_size[c]:
            unfilled_clusters.remove(c)
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
            size[nearest[p]] += 1
            assert size[nearest[p]] <= max_size[nearest[p]], (
                f"{size[nearest[p]]} !<= {max_size[nearest[p]]}")
            if size[nearest[p]] >= max_size[nearest[p]]:
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
    indices = [np.flatnonzero(labels == c) for c in range(num_cluster)]
    indices.sort(key=len, reverse=True)
    return indices


def _k_means_objective(population, labels, num_cluster):
    """Return the weighted sum of the generalized variance of each cluster."""
    xp = cp.get_array_module(population)
    cost = 0
    for c in range(num_cluster):
        weight = xp.sum(labels == c)
        if weight > 1:
            cost += weight * abs(
                xp.linalg.det(xp.cov(
                    population[labels == c],
                    rowvar=False,
                )))
    return cost


def _assert_cluster_is_full(labels, c, size):
    xp = cp.get_array_module(labels)
    assert size == np.sum(labels == c), ('All clusters should be full, but '
                                         f'cluster {c} had '
                                         f'{xp.sum(labels == c)} points '
                                         f'when it should have {size}.')


def cluster_wobbly_center(*args, **kwargs):
    import warnings
    warnings.warn(
        'tike.random.cluster_wobbly_center is depreacted. '
        'Use tike.cluster.wobbly_center instead.',
        DeprecationWarning,
    )
    return wobbly_center(*args, **kwargs)


def cluster_compact(*args, **kwargs):
    import warnings
    warnings.warn(
        'tike.random.cluster_compact is depreacted. '
        'Use tike.cluster.compact instead.',
        DeprecationWarning,
    )
    return compact(*args, **kwargs)


def _batch_ends(
    num_batch: int,
    size: int,
    index: int,
) -> typing.Tuple[int, int]:
    batch_size = size // num_batch
    remainder = size % num_batch
    lo = batch_size * index + min(remainder, index)
    hi = lo + batch_size + (1 if index < remainder else 0)
    return (lo, hi)
