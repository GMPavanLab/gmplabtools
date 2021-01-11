import pytest

import numpy as np

from gmplabtools.analysis import DataSampler


class TestDataSampler:

    def test_random_sample(self):
        # given:
        x = np.random.normal(0, 1, size=(1000, 10))

        # when:
        sampler = DataSampler(method="minkowski")
        result, indices = sampler.random_sample(x, 10)

        # then:
        assert result.shape[0] == 10

    @pytest.mark.parametrize("method", ["minkowski", "cosine", "global_mahalanobis", "local_mahalanobis"])
    def test_minmax(self, method):
        # given:
        x = np.random.normal(0, 1, size=(1000, 10))

        # when:
        sampler = DataSampler(method=method)
        result, indices = sampler.minmax_sample(x, 10)

        # then:
        assert result.shape[0] == 10

    def test_get_indices_dedup(self):
        # given:
        x = np.random.normal(0, 1, size=(10, 10))
        y = np.array([1] * 10, dtype=float)
        x = np.vstack([x, y])

        # when:
        indices = DataSampler.get_indices(x, True)

        # then:
        assert indices.shape[0] == 11

    def test_get_indices_not_dedup(self):
        # given:
        x = np.random.normal(0, 1, size=(100, 10))

        # when:
        indices = DataSampler.get_indices(x, False)

        # then:
        assert indices.shape[0] > 11