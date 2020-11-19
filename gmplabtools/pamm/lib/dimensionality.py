import logging

import numpy as np
from scipy.special import gamma

from gmplabtools.pamm.lib.tools import oracle_shrinkage


logging.getLogger(__name__).setLevel(level=logging.INFO)


class Dimension:
    """
    This class calculate a local covariance matrix for every points in a dataset.
    """
    def __init__(self, data, fs):
        self.fs = fs
        self.data = data
        self.total_Q = None
        self.sigma = None
        self.localisation = None

        self.set_values()

    def set_values(self):
        self.total_Q = np.cov(self.data.T)
        self.sigma = np.trace(self.total_Q) ** 0.5 * self.fs
        return self

    @staticmethod
    def gauss_local(y, x, sigma):
        """
        Calculate the local covariance matrix.
        """
        return np.exp(-0.5 * np.linalg.norm(x - y, axis=1) ** 2 / sigma**2)

    @staticmethod
    def local_cov(y, x, sigma):
        """
        Calculate the local covariance matrix.

        The coviariance bit could also be done as np.cov(X.T, weights=ui)
        """
        ui = Dimension.gauss_local(y, x, sigma)
        Ni = np.sum(ui)
        cov = np.empty((x.shape[1], x.shape[1])) * 0
        for i in range(x.shape[1]):
            for j in range(x.shape[1]):
                k = (x[:,i] - np.dot(x[:,i], ui) / Ni)
                l = (x[:,j] - np.dot(x[:,j], ui) / Ni)
                cov[i, j] = (np.dot(k * l,  ui) / Ni)

        return oracle_shrinkage(cov, x.shape[0]), ui, Ni

    def get_cov(self, i):
        """
        Calculate the covariance matrix.
        """
        values = Dimension.local_cov(self.data[i], self.data, self.sigma)
        self.localisation = values[1]
        return values[0]

    def local_dimension(self, i):
        return Dimension.cov_dim(self.get_cov(i))
    
    @staticmethod
    def cov_dim(cov):
        """
        Calculate the effective dimension given a covariance matrix.
        """
        eigen = np.linalg.eigvals(cov)
        p = eigen / np.sum(np.abs(eigen))
        return np.exp(- np.sum(p * np.log(p)))


class DataSampler:
    """
    Sample data according to a specified scheme.
    """
    def __init__(self, method="minkowski", norm=2):
        self.method = method
        self.norm = norm

    def __repr__(self):
        return "DataSampler(method={}, norm={})".format(self.method, self.norm)

    @staticmethod
    def inverse_cov(cov, n):
        return np.linalg.inv(oracle_shrinkage(cov, n))

    @property
    def distance_metric(self):
        return {'minkowski': DataSampler.minkowski,
                'cosine': DataSampler.cosine,
                'global_mahalanobis': DataSampler.global_mahalanobis,
                'local_mahalanobis': DataSampler.local_mahalanobis}[self.method]

    @staticmethod
    def minkowski(x, y, norm, **kwargs):
        """
        Vectorized Minkovsky distance of given norm.

        Args:
            x: Array of value fro which to calculate the distance to.
            y: Reference array.

        Returns:
            Array of distances from y.
        """
        return (np.sum(np.abs(x - y) ** norm, axis=1)) ** (1 / norm)

    @staticmethod
    def cosine(x, y, **kwargs):
        """
        Vectorized cosine distance.

        Args:
            x: Array of value fro which to calculate the distance to.
            y: Reference array.

        Returns:
            Array of distances from y.
        """
        return 1. - np.sum(x * y, axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y))

    @staticmethod
    def global_mahalanobis(x, y, cov_inv, **kwargs):
        """
        Vectorized Global mahalanobis distance.

        Args:
            x: Array of value fro which to calculate the distance to.
            y: Reference array.
            Qinv: Inverse of covariance matrix

        Returns:
            Array of distances from y.
        """
        return np.sqrt(np.sum((x - y) * np.matmul(x - y, cov_inv), axis=1))

    @staticmethod
    def local_mahalanobis(x, y, **kwargs):
        """
        Vectorized Local mahalanobis distance.

        Args:
            x: Array of value fro which to calculate the distance to.
            y: Reference array.
            Qinv: Inverse of covariance matrix

        Returns:
            Array of distances from y.
        """
        d = x.shape[1]
        dist = np.linalg.norm(x - y, axis=1)
        w = dist / np.pi ** (d / 2.) / gamma(d / 2. + 1.) * dist ** d
        i = np.argmax(np.where(x == y, 1, 0))
        # w[i - 1] = np.max(w[~np.isnan(w)])

        # oracle shrinkage of local covariance to avoid singular matrices
        local_cov_inv = DataSampler.inverse_cov(np.cov(x.T, aweights=w), x.shape[0])
        return np.sqrt(np.sum((x - y) * np.matmul(x - y, local_cov_inv), axis=1))

    @staticmethod
    def random_sample(x, n, keep_duplicates=False):
        """
        Generate a random sample of the dataset.
        Args:
            x: Original dataset.
            n: Size of the sample
            keep_duplicates: Whether to keep duplicated values.

        Returns:
            Sampled dataset.
        """
        # retrive index of unique values
        _, idxs = np.unique(list(map(str, x)), return_index=True)
        if not keep_duplicates:
            x = x[idxs]
            np.random.shuffle(idxs)
            idxs = idxs[:n]
            return x[idxs], idxs
        else:
            idxs = np.arange(x.shape[0])
            np.random.shuffle(idxs)
            idxs = idxs[:n]
            return x[idxs], idxs

    def minmax_sample(self, x, n, dedub=False):
        """
        Generate a random sample of the dataset using a minmax strategy.

        Args:
            x: Original dataset.
            n: Size of the sample
            dedup: De-duplicata datapoints

        Returns:
            Sampled dataset, index wrt origin dataset.
        """
        d = x.shape[1]
        # retrive index of unique values
        if dedub:
            _, idxs = np.unique(list(map(str, x)), return_index=True)
        else:
            idxs = np.arange(d)

        n = int(x.shape[0] ** 0.5) if n is None else n
        y = np.empty((n, d))

        cov_inv = DataSampler.inverse_cov(np.cov(x.T), x.shape[0])
        x = np.array(x)[idxs].reshape(-1, d)

        dmin = np.ones(x.shape[0]) * np.inf

        iy = np.random.randint(0, x.shape[0])
        y[0, :] = x[iy, :]
        # add point to new grid
        igrid = [idxs[iy]]
        scale = np.round(np.log10(n)) - 1
        steps = int(10 ** (1 if scale < 1 else min(4, scale)))

        for i in range(1, n):

            if (i + 1) % steps == 0:
                logging.info("Done {} steps of {}".format(i + 1, n))

            dx = self.distance_metric(x=x, y=y[i-1,:], norm=self.norm, cov_inv=cov_inv)
            dmin[dmin > dx] = dx[dmin > dx]
            iy = np.argmax(dmin)
            y[i, :] = x[iy, :]
            igrid.append(idxs[iy])

        return y, np.array(igrid, dtype=int)

