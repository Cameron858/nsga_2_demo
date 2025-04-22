import numpy as np
from typing import Iterable, Callable


def create_population(size: int, bounds: tuple) -> np.ndarray:
    """
    Creates an initial population for a genetic algorithm.

    Parameters
    ----------
    size : int
        The number of individuals in the population.
    bounds : tuple
        A tuple specifying the lower and upper bounds for the population values (bounds[0] is the lower bound, bounds[1] is the upper bound).

    Returns
    -------
    np.ndarray
        A numpy array of shape (size, 1) containing the generated population with values uniformly distributed within the specified bounds.
    """
    """"""
    return np.random.uniform(bounds[0], bounds[1], size=(size, 1))


def objective(population: np.ndarray, funcs: Iterable[Callable]) -> np.ndarray:
    """
    Compute the objective values for the given population and functions.

    Parameters
    ----------
    population : np.ndarray
        The current population with shape (size, n_variables).
    funcs : Iterable[Callable]
        A list of functions that take the population as input and return an array of objective values.

    Returns
    -------
    np.ndarray
        A numpy array of shape (size, n_objectives) containing the objective values for each individual in the population.
    """
    return np.hstack([f(population) for f in funcs])


def pareto_dominance(i1: np.ndarray, i2: np.ndarray) -> bool:
    """
    Checks if `i1` dominates `i2` in a Pareto sense.

    `i` -> `individual`

    Parameters
    ----------
    i1 : np.ndarray
        The objective values of the first individual.
    i2 : np.ndarray
        The objective values of the second individual.

    Returns
    -------
    bool
        True if `i1` dominates `i2`, False otherwise.
    """
    # i1 dominates i2 if it is at least as good in all objectives and strictly better in at least one
    # assumes all objectives are to minimise
    return np.all(i1 <= i2) and np.any(i1 < i2)
