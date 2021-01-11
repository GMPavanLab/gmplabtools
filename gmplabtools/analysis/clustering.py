import numpy as np
import scipy.sparse.csgraph as csg
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def calculate_adjacency(prob, clusters, bootstrap):
    """
    Calculated the adjacency matrix from a bootstrapped clustering procedure.

    Args:
        prob: Cluster probabilities for each gridpoint.
        clusters: Cluster label for each gridpoints.
        bootstrap: Bootstrap clustering data.

    Returns:
        Array of distances from y, clusters to gridpoints mapping
    """
    uc = np.unique(clusters)
    ncls = len(uc)
    nbs = len(bootstrap)
    nclsbs = np.zeros(nbs, int)

    # maximum number of cluster per bootstrap run
    for bs in np.arange(nbs):
        nclsbs[bs] = np.max(bootstrap[bs]) + 1

    # probability of each cluster and cluster indices
    lcls = []
    Qi = np.zeros(ncls)
    for i in np.arange(ncls):
        icls = np.where(clusters == uc[i])[0]
        Qi[i] = np.exp(prob[icls]).sum()
        lcls.append(icls)

    # probability of each cluster for every bootstrap run
    QA = np.zeros((nbs, max(nclsbs))) + 1e-6
    for bs in np.arange(nbs):
        for i in np.arange(nclsbs[bs]):
            icls = np.where(bootstrap[bs] == i)[0]
            QA[bs, i] = np.exp(prob[icls]).sum()

    # probability of each in global cluster i of being in bootstrap cluster j
    QAi = np.zeros((nbs, max(nclsbs), ncls)) + 1e-6
    for bs in np.arange(nbs):
        for i in np.arange(nclsbs[bs]):
            icls = np.where(bootstrap[bs] == i)[0]
            for j in np.arange(ncls):
                inter = np.intersect1d(icls, lcls[j])
                QAi[bs, i, j] = np.exp(prob[inter]).sum()

    # build adjacency
    nij = np.zeros((ncls, ncls))
    for i in np.arange(ncls):
        for j in np.arange(i + 1):
            tij = 0
            for bs in np.arange(nbs):
                for k in np.arange(nclsbs[bs]):
                    if QA[bs, k] != 0:
                        tij += QAi[bs, k, i] * QAi[bs, k, j] / QA[bs, k]
            nij[i, j] = tij / nbs

    nij /= np.exp(prob).sum()
    Qi /= np.exp(prob).sum()
    nnij = nij / np.sqrt(np.multiply.outer(Qi, Qi))
    return nnij, lcls


def merge(adjacency, cluster_mapping, threshold):
    """
    Marge clusters given a threshold of the adjecency matrix.

    Args:
        adjacency: Adjacency matrix.
        cluster_mapping: Cluster to gridpoint mapping.
        threshold: Distance threshold.

    Returns:
        Array of distances from y, cluster mapping.
    """
    N = len(sum(map(list, cluster_mapping), [])) # number of gridpoints
    imacro = np.ones(N, dtype=int) * -1

    # for the terms above a given threshold, find connected sub-graphs
    cij = adjacency > threshold
    cgraph = csg.csgraph_from_dense(cij, null_value=False)
    cc = csg.connected_components(cgraph)
    for i in np.arange(cc[0]):
        for j in np.arange(len(adjacency)):
            if cc[1][j] == i:
                imacro[cluster_mapping[j]] = i
    return imacro


def adjancency_dendrogram(adjacency, link="ward"):
    """
    This function returns a square-form distance matrix used for building a
    dendrogram based on the clusters' adjacency matrix
    """
    dist = np.zeros(adjacency.shape)
    for i in range(adjacency.shape[0]):
        for j in range(adjacency.shape[1]):
            if adjacency[i, j] > 0:
                dist[i, j] = -np.log(
                    adjacency[i, j] / np.sqrt(adjacency[i, i] * adjacency[j, j])
                )

    # copy lower triangular into upper triangular
    dist = np.tril(dist).T + dist

    # set furthest distance to max in distance matrix
    dist[dist == 0] = np.inf
    dist[dist == np.inf] = max(dist.flatten()[dist.flatten() != np.inf])

    # distance definition makes diagonal entries zero
    np.fill_diagonal(dist, 0)

    # single is the only way this distance matrix can be interpreted
    return linkage(squareform(dist), link)
