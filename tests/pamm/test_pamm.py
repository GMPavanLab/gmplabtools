from unittest import mock
import pytest

import numpy as np

from gmplabtools.pamm import Pamm, Gauss, PammGMM
from tests.conftest import InputDictFixture


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


class TestPammGMM:

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
        gmm = PammGMM(pk=pk, mean_k=mean_k, cov_k=cov_k)
        proba = gmm.predict_proba(x)

        # then:
        assert len(np.where(np.argmax(proba, axis=1) == 9)[0]) == 0


class TestPamm:

    @mock.patch('os.path.isfile', lambda x: True)
    def test_init(self, input_dict):
        # when:
        pamm = Pamm(input_dict)

        # then:
        assert pamm.dimension == 3

    @InputDictFixture(fspread=0.2)
    @mock.patch('os.path.isfile', lambda x: True)
    def test_format_fpoints_and_fspread(self, input_dict):
        # when:
        pamm = Pamm(input_dict)

        # then:
        with pytest.raises(ValueError, match="Must provide only"):
            pamm.format()

    @InputDictFixture(fpoints=None, fspread=None)
    @mock.patch('os.path.isfile', lambda x: True)
    def test_format_fpoints_and_fspread_missing(self, input_dict):
        # when:
        pamm = Pamm(input_dict)

        # then:
        with pytest.raises(ValueError, match="Must provide only"):
            pamm.format()

    @InputDictFixture(readgrid="some_gridfile")
    def test_grid_file_not_found(self, input_dict):
        # when:
        pamm = Pamm(input_dict)

        # then:
        with pytest.raises(ValueError, match="Grid file `some_gridfile` was not found"):
            pamm.format()