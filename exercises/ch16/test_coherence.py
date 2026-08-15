"""E-16.10.  How loose is D-16.2's union bound?  Measure it."""
import math

import numpy as np
import pytest  # noqa: F401

from exercises.ch16.solution import (largest_m_within, max_coherence,
                                     random_dictionary)


def test_random_dictionary_is_unit_norm():
    rng = np.random.default_rng(1610)
    U = random_dictionary(500, 64, rng)
    assert U.shape == (500, 64)
    assert np.allclose(np.linalg.norm(U, axis=1), 1.0)


def test_max_coherence_matches_bound():
    """The achieved m exceeds exp(d eps^2/4), and by a single-digit factor
    rather than the orders of magnitude folklore suggests."""
    rng = np.random.default_rng(7)
    for d, eps in ((128, 0.30), (256, 0.25), (512, 0.20)):
        m = largest_m_within(eps, d, rng)
        bound = math.exp(d * eps ** 2 / 4)
        assert m >= bound, "the bound is a lower bound on what is achievable"
        assert 2.0 < m / bound < 12.0
