from src.simple_regressor.metrics import *


class TestMetrics:
    def test_absolute_error():
        assert absolute_error(5, 3) == 2