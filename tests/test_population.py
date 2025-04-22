import pytest
from nsga_2.algo import create_population


@pytest.mark.parametrize(
    "size, bounds", [(10, (0, 1)), (1, (0, 1)), (100, (0, 1)), (100, (2, 5))]
)
def test_that_population_is_created_correctly(size, bounds):
    p = create_population(size, bounds)

    assert p.shape[0] == size
    assert (p >= bounds[1]).all
    assert (p >= bounds[0]).all
