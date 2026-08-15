"""E-16.11.  Interference grows as sqrt(k/d), and that is what sets the limit."""
import math

import numpy as np
import pytest

from exercises.ch16.solution import (interference_std, largest_k_separating,
                                     random_dictionary)

D, M = 1024, 4096


@pytest.mark.parametrize("k", [10, 50, 200])
def test_interference_std_is_sqrt_k_over_d(k):
    rng = np.random.default_rng(1611 + k)
    U = random_dictionary(M, D, rng)
    s = interference_std(U, k, 60, rng)
    assert s == pytest.approx(math.sqrt(k / D), rel=0.10)


def test_separation_matches_the_bound():
    """(16.14) with z^2 = 4 ln m is exactly the sparsity at which m
    simultaneous reads still separate."""
    rng = np.random.default_rng(5)
    U = random_dictionary(M, D, rng)
    z = 2 * math.sqrt(math.log(M))
    k = largest_k_separating(U, z, 40, rng)
    predicted = D / (4 * math.log(M))
    assert k == pytest.approx(predicted, rel=0.35)
