import pytest
from src.simple_regressor.metrics import *


class TestMetrics:

    @pytest.fixture
    def array_regression_fixture_1(self):
        return np.array([[5, 3, 9.5, -12, 0, 19], [3, 2, 7, -5, -4, 1]])

    @pytest.fixture
    def array_regression_fixture_2(self):
        return np.array([[2.5, 0.0, 2, 8], [3, -0.5, 2, 7]])

    @pytest.fixture
    def array_classification_proba_fixture_1(self):
        return np.array([[0.5, 0.8, 0.7, 0.9, 0.2], [0, 1, 1, 0, 0]])

    @pytest.fixture
    def array_classification_proba_fixture_2(self):
        return np.array([[0.9, 0.1, 0.2, 0.65], [1, 0, 0, 1]])

    @pytest.fixture
    def array_classification_label_fixture_1(self):
        return np.array([[1, 0, 0, 0], [1, 1, 0, 1]])

    @pytest.fixture
    def array_classification_label_fixture_2(self):
        return np.array([[0, 0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 0, 0, 1]])

    @pytest.mark.parametrize("pred_y, expec_y, expected", [(5, 3, 2),
        (3, 5, 2),
        (-1, -1, 0),
        (0, 0, 0),
        (5, -3, 8),
        (-5, 3, 8)])
    def test_absolute_error(self, pred_y, expec_y, expected):
        assert absolute_error(pred_y, expec_y) == expected

    @pytest.mark.parametrize("pred_y, expec_y, expected", [(5, 3, 4),
        (3, 5, 4),
        (-1, -1, 0),
        (0, 0, 0),
        (5, -3, 64),
        (-5, 3, 64)])
    def test_squared_error(self, pred_y, expec_y, expected):
        assert squared_error(pred_y, expec_y) == expected

    @pytest.mark.parametrize('fixt, expected', [
        ('array_regression_fixture_1', 66.70833333333333),
        ('array_regression_fixture_2', 0.375)])
    def test_mean_squared_error(self, fixt, expected, request):
        y = request.getfixturevalue(fixt)
        assert mean_squared_error(y[0], y[1]) == pytest.approx(expected)

    @pytest.mark.parametrize('fixt, expected', [
        ('array_regression_fixture_1', 5.75),
        ('array_regression_fixture_2', 0.5)])
    def test_mean_absolute_error(self, fixt, expected, request):
        y = request.getfixturevalue(fixt)
        assert mean_absolute_error(y[0], y[1]) == pytest.approx(expected)

    @pytest.mark.parametrize('fixt, expected', [
        ('array_regression_fixture_1', -2.949835526315789),
        ('array_regression_fixture_2', 0.9486081370449679)])
    def test_r2(self, fixt, expected, request):
        y = request.getfixturevalue(fixt)
        assert r2(y[0], y[1]) == pytest.approx(expected)

    @pytest.mark.parametrize('fixt, expected', [
        ('array_classification_proba_fixture_1', 0.7597388640242285),
        ('array_classification_proba_fixture_2', 0.21616187468057912)])
    def test_cross_entropy(self, fixt, expected, request):
        y = request.getfixturevalue(fixt)
        assert cross_entropy(y[0], y[1]) == pytest.approx(expected)
    
    @pytest.mark.parametrize('fixt, expected', [
        ('array_classification_label_fixture_1', 0.5),
        ('array_classification_label_fixture_2', 0.714285714)])
    def test_accuracy(self, fixt, expected, request):
        y = request.getfixturevalue(fixt)
        assert accuracy(y[0], y[1]) == pytest.approx(expected)