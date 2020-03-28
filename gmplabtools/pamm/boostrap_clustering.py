import numpy as np
import scipy.sparse.csgraph as csg


def adjacency(prob, clusters, boot):
    """
    Calculated the adjacency matrix from a boostrapped clustering procedure.

    Args:
        prob: Cluster probabilities for each gridpoint.
        clusters: Cluster label for each gridpoints.
        boot: Bootstrap clustering data.
    Returns:
        Array of distances from y, clusters to gridpoints mapping
    """
    uc = np.unique(clusters)
    ncls = len(uc)
    nbs = len(boot)
    nclsbs = np.zeros(nbs, int)

    # maximum number of cluster per bootstrap run
    for bs in np.arange(nbs):
        nclsbs[bs] = np.max(boot[bs]) + 1

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
            icls = np.where(boot[bs] == i)[0]
            QA[bs, i] = np.exp(prob[icls]).sum()

    # probability of each in global cluster i of being in boostrap cluster j
    QAi = np.zeros((nbs, max(nclsbs), ncls)) + 1e-6
    for bs in np.arange(nbs):
        for i in np.arange(nclsbs[bs]):
            icls = np.where(boot[bs] == i)[0]
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
        threshold: Distance thresold.

    Returns:
        Array of distances from y, cluster mapping.
    """
    N = sum(map(list, cluster_mapping), []) # number of gridpoints
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