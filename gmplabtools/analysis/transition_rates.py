import numpy as np


class ClusterRates:
    """
    Class that return clustering transition rate matrix
    """
    def __init__(self, n, method="label"):
        """
        Args:
            n: number of components of each individual frames.
            method: whether to calculate transition rate using cluster labels or cluster assignment
                probabilities.
        """
        self.n = n
        self.method = method
        self.clusters = None
        self.n_cluster = None

    def _select_method(self):
        return {
            "label": ClusterRates.matrix_update_cluster,
            "proba": ClusterRates.matrix_update_proba
        }[self.method]

    def _get_dimension(self, x):
        if self.method == "label":
            clusters = np.unique(x)
        else:
            clusters = np.arange(x.shape[1], dtype=int)
        return clusters, len(clusters)

    def calculate_matrix(self, x):
        """
        Calculates the transition matrix using cluster labels

        Args:
            x: Array of cluster labels for each molecule and timeframe.

        Returns:
            transition_matrix: Cluster transition matrix
        """
        self.clusters, self.n_clusters = self._get_dimension(x)

        if self.method == "label":
            _x = x.reshape((int(x.shape[0] / self.n), self.n))
        else:
            _x = x.reshape((int(x.shape[0] / self.n), self.n, self.n_clusters))

        transition_matrix = np.zeros((self.n_clusters, self.n_clusters))

        for i, frame in enumerate(_x[1:, :]):
            transition_matrix += self._select_method()(_x[i, :],
                                                       frame,
                                                       self.clusters,
                                                       self.n)

        transition_matrix = np.nan_to_num(
            transition_matrix / transition_matrix.sum(axis=1)[:, np.newaxis],
            0
        )
        return transition_matrix

    @staticmethod
    def matrix_update_cluster(v, w, clusters, n):
        """
        Calculates transition matrix updates for two consecutive frames

        Args:
            v: Frame of trajectory.
            w: Subsequent frame of trajectory.
            clusters: Total sequence of clusters.
            n: number of molecules.

        Returns:
            update: transition matrix update.
        """
        update = np.zeros((len(clusters), len(clusters)), dtype=int)
        for i, c1 in enumerate(clusters):
            y = w[v == c1]
            for j, c2 in enumerate(clusters):
                update[i, j] = len(np.where(y == c2)[0])
        return update

    @staticmethod
    def matrix_update_proba(v, w, clusters, n):
        """
        Calculates transition matrix updates for two consecutive frames
        using cluster probabilities.

        Args:
            v: Frame of trajectory.
            w: Subsequent frame of trajectory.
            clusters: Total sequence of clusters.
            n: number of molecules.

        Returns:
            update: transition matrix update.
        """
        n_cl = len(clusters)
        update = np.zeros((n_cl, n_cl), dtype=float)
        for i, c1 in enumerate(range(n)):
            update += v[[i]].T * w[[i]]
        return update