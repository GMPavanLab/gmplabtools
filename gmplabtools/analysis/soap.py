import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering


class BaseSoapClustering:
    """Base to use perform clustering on SOAP descriptors using estimator based on a distance
     matrix.
     """

    def __init__(self, n_neighbors=1):
        self._knn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=BaseSoapClustering.soap_distance
        )
        self._clustering = None
        self.model = None

    @staticmethod
    def soap_kernel(x, y):
        u = np.dot(x, x)
        v = np.dot(y, y)
        return np.dot(x, y) / (u * v) ** 0.5

    @staticmethod
    def soap_distance(x, y):
        kernel = BaseSoapClustering.soap_kernel(x, y)
        d = (2 - 2 * kernel) ** 0.5
        return d if not np.isnan(d) else 0

    @property
    def distance(self):
        return squareform(self._distance)

    def get_dendrogram(self, link="average"):
        return linkage(self.distance, link)

    def _fit(self, x):
        self._distance = pdist(x, BaseSoapClustering.soap_distance)
        self._knn.fit(x)

    def fit(self, x):
        return self

    def predict(self, x):
        if x.shape[0] != self.distance.shape[0]:  # if you are predicting out of sample
            _, i = self._knn.kneighbors(x)
            labels = self.model.labels_[i[:, :1]]
        else:
            labels = self.model.labels_
        return labels.reshape(-1, )


class AggClustering(BaseSoapClustering):
    """Class which implements clustering on SOAP descriptors using AgglomerativeClustering.
     """
    def fit(self, x, distance_threshold, linkage="average", n_clusters=None, **kwargs):
        self._fit(x)
        self.model = AgglomerativeClustering(
            affinity="precomputed",
            linkage=linkage,
            distance_threshold=distance_threshold,
            n_clusters=n_clusters,
            **kwargs
        ).fit(self.distance)
        return self


class AffClustering(BaseSoapClustering):
    """Class which implements clustering on SOAP descriptors using AffinityPropagation.
     """
    def fit(self, x, **kwargs):
        self._fit(x)
        self.model = AffinityPropagation(
            affinity='precomputed',
            **kwargs
        ).fit(self.distance)
        return self
