import numpy as np


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
