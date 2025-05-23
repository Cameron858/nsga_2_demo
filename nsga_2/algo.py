import numpy as np
from typing import Iterable, Callable


def create_population(size: int, bounds: tuple) -> np.ndarray:
    """
    Create an initial population for a genetic algorithm.

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
    Check if `i1` dominates `i2` in a Pareto sense.

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
    Assign Pareto fronts to a population based on objective values.

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


def flatten_fronts(p_obj: np.ndarray, fronts: dict[set[int]]) -> np.ndarray:
    """
    Assign a front number to each individual in the population based on the provided fronts.

    Parameters
    ----------
    p_obj : np.ndarray
        A 2D array where each row represents an individual and each column represents an objective value.
    fronts : dict of set of int
        A dictionary where the keys are front numbers (starting from 0) and the values are sets of indices
        corresponding to individuals in that front.

    Returns
    -------
    np.ndarray
        A 1D array where each element corresponds to the front number assigned to the respective individual
        in the population.
    """
    f = np.zeros(p_obj.shape[0])
    for front, members in fronts.items():
        f[list(members)] = front
    return f


def calculate_crowding_distance(p_obj: np.ndarray) -> np.ndarray:
    """
    Calculate the crowding distance for each individual.

    Parameters
    ----------
    p_obj : np.ndarray
        A (N, M) array where N is the number of individuals and M is the number of objectives.

    Returns
    -------
    np.ndarray
        An array of crowding distances for each individual.
    """
    crowding_distances = np.zeros(p_obj.shape[0])

    for m in range(p_obj.shape[1]):  # For each objective
        m_values = p_obj[:, m]
        m_range = m_values.max() - m_values.min()

        # Sort m_values and get sorted indices
        m_sorted_indices = np.argsort(m_values)

        # Set the crowding distance for the boundary points to inf
        boundary_indices = m_sorted_indices[[0, -1]]
        crowding_distances[boundary_indices] = np.inf

        # Update the in-between points
        for i in range(1, m_sorted_indices.shape[0] - 1):
            prev_i = m_sorted_indices[i - 1]
            next_i = m_sorted_indices[i + 1]

            increment = (m_values[next_i] - m_values[prev_i]) / m_range
            crowding_distances[m_sorted_indices[i]] += increment

    return crowding_distances


def tournament_select(
    p_obj: np.ndarray, fronts: dict[set[int]], crowding_distances: np.ndarray
) -> int:
    """
    Perform tournament selection based on Pareto fronts and crowding distances.

    Parameters
    ----------
    p_obj : np.ndarray
        The population objective values, where each row corresponds to an individual
        and each column corresponds to an objective.
    fronts : dict[set[int]]
        A list of Pareto fronts, where each front is a list of indices corresponding
        to individuals in the population.
    crowding_distances : np.ndarray
        An array of crowding distances for each individual in the population.

    Returns
    -------
    int
        The index of the selected individual from the population.
    """
    fronts = flatten_fronts(p_obj, fronts)

    members = np.array(list(zip(fronts, crowding_distances)))
    selected_i = np.random.choice(p_obj.shape[0], 2, replace=False)
    selected = members[selected_i]

    # sort by front (ascending), then by crowding distance (descending, so we negate)
    winner_rel_i = np.lexsort((-selected[:, 1], selected[:, 0]))[0]
    winner_abs_i = selected_i[winner_rel_i]

    return winner_abs_i


def sbx_crossover(
    p1: np.ndarray, p2: np.ndarray, eta=20
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Simulated Binary Crossover (SBX) between two parent solutions.

    Parameters
    ----------
    p1 : np.ndarray
        The first parent solution.
    p2 : np.ndarray
        The second parent solution.
    eta : int, optional
        The distribution index, which controls the spread of offspring solutions.
        Higher values result in offspring closer to the parents. Default is 20.

    Returns
    -------
    tuple[np.ndarray]
        A tuple containing two offspring solutions as NumPy arrays.
    """
    u = np.random.uniform(size=p1.shape)
    beta = np.where(
        u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1))
    )

    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)

    return c1, c2


def polynomial_mutation(x: np.ndarray, bounds: tuple, eta=5) -> np.ndarray:
    """
    Perform polynomial mutation on a given array.

    Parameters
    ----------
    x : np.ndarray
        The input array to be mutated.
    bounds : tuple
        A tuple specifying the lower and upper bounds for the mutation
        (bounds[0] is the lower bound, bounds[1] is the upper bound).
    eta : int, optional
        The distribution index that controls the extent of the mutation.
        Higher values of `eta` result in smaller mutations. Default is 5.

    Returns
    -------
    np.ndarray
        The mutated array, with values clipped to the specified bounds.
    """
    u = np.random.uniform(size=x.shape)
    # pertubation
    exponent = 1 / (1 + eta)
    delta = np.where(
        u < 0.5, ((2 * u) ** exponent) - 1, 1 - ((2 * (1 - u)) ** exponent)
    )
    x_mutated = x + (delta * (bounds[1] - bounds[0]))
    x_mutated = np.clip(x_mutated, bounds[0], bounds[1])

    return x_mutated


def generate_offspring(
    p_obj: np.ndarray,
    p: np.ndarray,
    fronts: np.ndarray,
    crowding_distances: np.ndarray,
    bounds: tuple,
    sbx_prob: float = 0.9,
    mutation_prob: float = 0.05,
) -> np.ndarray:
    """
    Generate offspring using tournament selection, SBX crossover, and polynomial mutation.

    Parameters
    ----------
    p_obj : np.ndarray
        Objective values of the current population.
    p : np.ndarray
        Current population.
    fronts : np.ndarray
        Array indicating the front number for each individual.
    crowding_distances : np.ndarray
        Crowding distances for each individual.
    bounds : tuple
        Tuple containing lower and upper bounds (both np.ndarray) for variables.
    sbx_prob : float, optional
        Probability of performing SBX crossover, by default 0.9.
    mutation_prob : float, optional
        Probability of applying polynomial mutation to each child, by default 0.05.

    Returns
    -------
    np.ndarray
        New offspring population of the same size as the original.
    """
    # create mating pool
    mating_pool = []
    while len(mating_pool) < p.shape[0]:

        # select a winner
        winner_i = tournament_select(p_obj, fronts, crowding_distances)
        mating_pool.append(p[winner_i])

    assert len(mating_pool) == p.shape[0]

    # breed in pairs
    Q = []
    for i in range(0, p.shape[0], 2):

        p1 = mating_pool[i]
        p2 = mating_pool[i + 1]

        # roll for crossover, else propagate parents as children
        if np.random.rand() < sbx_prob:
            children = sbx_crossover(p1, p2)
        else:
            children = p1, p2

        # independantly mutate children
        children = [
            polynomial_mutation(c, bounds) if np.random.rand() < mutation_prob else c
            for c in children
        ]

        Q.extend(children)

    return np.array(Q)
