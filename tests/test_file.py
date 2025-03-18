import numpy as np
import habmot


def test_version():
    assert habmot.__version__ == "0.1.0"


def test_adder():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    result = habmot.adder(a, b)
    assert np.all(result == np.array([5, 7, 9]))
