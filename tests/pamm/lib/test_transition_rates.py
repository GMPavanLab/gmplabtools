import numpy as np

from gmplabtools.pamm.lib.transition_rates import ClusterRates


class TestClusterRates:

    def test__get_dimension(self):
        # given:
        x = np.random.randint(3, size=(100, 5))

        # when:
        rates = ClusterRates(10, method='label')

        # then:
        clusters, n_cluster = rates._get_dimension(x)
        np.testing.assert_almost_equal(np.array([0, 1, 2]), clusters)
        assert n_cluster == 3

        # when:
        rates = ClusterRates(10, method='proba')

        # then:
        clusters, n_cluster = rates._get_dimension(x)
        np.testing.assert_almost_equal(np.array([0, 1, 2, 3, 4]), clusters)
        assert n_cluster == 5

    def test_method_label(self):
        # given:
        x = np.array([1, 2, 1, 1, 2, 2])

        # when:
        rates = ClusterRates(3, method='label')

        # then:
        trans_matrix = rates.calculate_matrix(x)
        np.testing.assert_almost_equal(np.array([[.5, .5], [0, 1.]]), trans_matrix)

    def test_method_proba(self):
        # given:
        x = np.array([[0.8, 0.2], [0.4, 0.6], [0.8, 0.2], [1, 0.], [0, 1.], [0., 1.]])

        # when:
        rates = ClusterRates(3, method='proba')

        # then:
        trans_matrix = rates.calculate_matrix(x)
        np.testing.assert_almost_equal(np.array([[.4, .6], [0.2, 0.8]]), trans_matrix)