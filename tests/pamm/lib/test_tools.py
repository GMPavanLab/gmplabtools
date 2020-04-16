import numpy as np
from gmplabtools.pamm.lib.tools import Gauss, GMMPredict


class TestGauss:

    def test_null_covariance(self):
        # given:
        means = np.random.normal(0, 1, size=10)
        cov = np.empty((10, 10))
        cov[:] = np.nan

        # when:
        x = np.random.dirichlet([1] * 10, size=1)
        gauss = Gauss(mean=means, cov=cov)

        # then:
        assert gauss.dim == 10
        assert gauss._valid_covariance == False
        assert np.isclose(gauss.pdf(x), 0)


class TestGMMPredict:

    def test_null_covariance(self):
        # given:
        pk = [i for i in np.random.dirichlet([1] * 10, size=1)[0]]
        mean_k = [np.random.normal(0, 1, size=10) for _ in range(10)]
        cov_k = [np.diag(np.random.choice([1,2,3, 4], size=10)) for _ in range(9)]
        null = np.empty((10, 10))
        null[:] = np.nan
        cov_k.append(null)

        # when:
        x = np.random.dirichlet([1] * 10, size=[1000])
        gmm = GMMPredict(pk=pk, mean_k=mean_k, cov_k=cov_k)
        proba = gmm.predict_proba(x)

        # then:
        assert len(np.where(np.argmax(proba, axis=1) == 9)[0]) == 0