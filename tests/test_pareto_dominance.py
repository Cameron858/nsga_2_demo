import pytest
from nsga_2.algo import pareto_dominance
import numpy as np


def test_pareto_dominance_dominates():
    i1 = np.array([1, 2, 3])
    i2 = np.array([2, 3, 4])
    assert pareto_dominance(i1, i2)


def test_pareto_dominance_not_dominates():
    i1 = np.array([2, 3, 4])
    i2 = np.array([1, 2, 3])
    assert not pareto_dominance(i1, i2)


def test_pareto_dominance_equal():
    i1 = np.array([1, 2, 3])
    i2 = np.array([1, 2, 3])
    assert not pareto_dominance(i1, i2)


def test_pareto_dominance_partial_dominance():
    i1 = np.array([1, 3, 2])
    i2 = np.array([2, 2, 3])
    assert not pareto_dominance(i1, i2)


def test_pareto_dominance_strictly_better_in_one():
    i1 = np.array([1, 2, 3])
    i2 = np.array([1, 2, 4])
    assert pareto_dominance(i1, i2)
