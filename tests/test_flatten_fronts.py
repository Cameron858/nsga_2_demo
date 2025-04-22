import numpy as np
from nsga_2.algo import flatten_fronts


def test_flatten_fronts_single_front():
    p_obj = np.array([[1, 2], [2, 3], [3, 4]])
    fronts = {1: {0, 1, 2}}
    expected = np.array([1, 1, 1])
    result = flatten_fronts(p_obj, fronts)
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"


def test_flatten_fronts_multiple_fronts():
    p_obj = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    fronts = {1: {0, 1}, 2: {2}, 3: {3}}
    expected = np.array([1, 1, 2, 3])
    result = flatten_fronts(p_obj, fronts)
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"


def test_flatten_fronts_empty_population():
    p_obj = np.array([])
    fronts = {}
    expected = np.array([])
    result = flatten_fronts(p_obj, fronts)
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"


def test_flatten_fronts_disjoint_fronts():
    p_obj = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    fronts = {1: {0}, 2: {1, 2}, 3: {3}}
    expected = np.array([1, 2, 2, 3])
    result = flatten_fronts(p_obj, fronts)
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"


def test_flatten_fronts_unordered_fronts():
    p_obj = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    fronts = {2: {1, 2}, 1: {0}, 3: {3}}
    expected = np.array([1, 2, 2, 3])
    result = flatten_fronts(p_obj, fronts)
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"
