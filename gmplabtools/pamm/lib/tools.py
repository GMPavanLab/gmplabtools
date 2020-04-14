import os
import logging

import numpy as np


logging.getLogger(__name__).setLevel(level=logging.INFO)


def oracle_shrinkage(cov, size):
    """
    Perform covariance matrix regularisation using oracle shrinkage.

    Args:
        cov: Covariance matrix.
        size: Original dataset size.

    Returns:
        Regularized covariance.
    """
    d = 1 if len(cov.shape) < 2 else cov.shape[1]

    # apply oracle approximating shrinkage alogorithm on local Q
    num = (1 - 2 / d) * np.trace(cov ** 2) + np.trace(cov) ** 2
    den = (size + 1. - 2 / d) * np.trace(cov ** 2) - np.trace(cov) ** 2 / d
    b = min(1, num / den)

    # regularized local covariance matrix for grid point
    return (1. - b) * cov + b * np.trace(cov) * np.identity(int(d)) / d


class Gauss:
    """
    Class that wraps a normally distibuted random variable.
    """
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self._cov_det = self.cov_det()

    @property
    def dim(self):
        return len(self.mean)

    def cov_det(self):
        det = np.linalg.det(self.cov)
        return max(det, 1E-20)

    def logpdf(self, x):
        shift = x - self.mean
        a = -np.log(((2 * np.pi) ** self.dim * self._cov_det) ** 0.5)
        b = np.dot(np.dot(shift, np.linalg.inv(self.cov)), shift)
        return a - 0.5 * b

    def pdf(self, x):
        return np.exp(self.logpdf(x))


class GMMPredict:
    """
    This class received the GMM parameters from pamm and is used to predict on
    other datesets.
    """
    def __init__(self, pk, mean_k, cov_k):
        self.pk = pk
        self.mean_k = mean_k
        self.cov_k = cov_k
        self.mixtures = [Gauss(m, c) for m, c in zip(mean_k, cov_k)]

    def _predict_proba(self, x):
        log_prob = [mixture.logpdf(x) for mixture in self.mixtures]
        prob = np.array([np.exp(log_p) * self.pk[i] for i, log_p in enumerate(log_prob)])
        return prob

    def predict_proba(self, x):
        """
        Predict cluster probabilities for a dataset.
        """
        prob = np.apply_along_axis(self._predict_proba, 1, x)
        return  prob / np.sum(prob, axis=1).reshape((-1, 1))

    @classmethod
    def read_clusters(cls, filename):
        """
        Read pamm output parameter and return an instance of GMMPredict class.
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError("File {} not found.".format(filename))
        else:
            pk, means, cov, = [], [], []
            skipped = []
            for i, line in enumerate(open(filename, 'r').readlines()):
                if i == 2:
                    D, n_cluster = map(int, line.split())
                if i > 2:
                    values = list(map(float, line.split()))
                    p, m, c = values[0], np.array(values[1:1 + D]), np.array(values[1 + D:]).reshape((D, D))
                    if np.isnan(c).sum() == 0:
                        pk.append(p)
                        means.append(m)
                        cov.append(oracle_shrinkage(c, D))
                    else:
                        skipped.append(str(i - 2))
            if skipped:
                logging.info("Skipped {} clusters: {}".format(len(skipped), ", ".join(skipped)))
            return cls(pk, means, cov)