import numpy as np
from gmplabtools.pamm.lib.tools import Gauss


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
