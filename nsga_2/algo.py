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


def assign_fronts(p_obj: np.ndarray) -> dict[int, set[int]]:
    """
    Assigns Pareto fronts to a population based on objective values.

    Parameters
    ----------
    p_obj : np.ndarray
        A (N, M) array where N is the number of individuals and M is the number of objectives.

    Returns
    -------
    dict[int, set[int]]
        A dictionary mapping front number to sets of individual indices belonging to that front.
    """
    # initialise first front
    F = {1: set()}

    # S -> "dominates" counter
    S = {i: set() for i in range(p_obj.shape[0])}

    # n -> "domination" counter i.e. "dominated by"
    n = np.zeros(p_obj.shape[0])

    # find members of F1
    for p_idx, p in enumerate(p_obj):

        for q_idx, q in enumerate(p_obj):

            # do not compare same individuals
            if p_idx == q_idx:
                continue

            # if p dominates q, add q to the set of solutions dominated by p
            if pareto_dominance(p, q):
                S[p_idx].add(q_idx)
            # if q dominated p, increase the domination counter of p
            elif pareto_dominance(q, p):
                n[p_idx] += 1

        if n[p_idx] == 0:
            F[1].add(p_idx)

    # populate the rest of the fronts
    i = 1
    while len(F[i]) != 0:

        # init members of next front
        Q = set()

        for p in F[i]:
            for q in S[p]:
                n[q] -= 1

                if n[q] == 0:
                    Q.add(q)

        # avoid empty last set
        if not Q:
            break
        i += 1
        F[i] = Q

    # check no members have been left out
    assert p_obj.shape[0] == len(set().union(*F.values()))

    return F
