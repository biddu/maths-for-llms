"""E-14.13.  Speculative decoding emits exactly p, and the plausible bug does not.

This is the chapter's signature exercise.  D-14.3 is a theorem, so the correct
implementation is not merely close to p, it IS p, and no sample size will
reject it.  The bug is the one everyone writes first: on rejection, resample
from p instead of from the residual.  Its law is min(p, q) + TV(p, q) * p,
which is derivable in two lines and detectable in one test.
"""
import numpy as np
import pytest
from scipy.stats import chisquare

from exercises.ch14.solution import (expected_tokens, speculative_emit,
                                     speculative_emit_broken)

V = 50
N = 200_000


def _pair(seed=1413):
    """A realistic pair: q is p seen through a noisy channel, giving a total
    variation distance near 0.18 and an acceptance rate near A-14.1's 0.8."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.full(V, 0.8))
    z = np.log(p) + rng.standard_normal(V) * 0.5
    q = np.exp(z - z.max())
    return p, q / q.sum()


def test_output_distribution_matches_target():
    p, q = _pair()
    rng = np.random.default_rng(0)
    counts = np.bincount(speculative_emit(p, q, rng, N), minlength=V)
    assert counts.sum() == N
    assert chisquare(counts, p * N).pvalue > 0.01


def test_broken_residual_is_detected():
    p, q = _pair()
    rng = np.random.default_rng(0)
    counts = np.bincount(speculative_emit_broken(p, q, rng, N), minlength=V)
    assert chisquare(counts, p * N).pvalue < 1e-6


def test_broken_law_is_the_derivable_one():
    """min(p, q) + TV * p, to two decimal places of the empirical frequency."""
    p, q = _pair()
    rng = np.random.default_rng(1)
    freq = np.bincount(speculative_emit_broken(p, q, rng, 400_000),
                       minlength=V) / 400_000
    tv = 0.5 * np.abs(p - q).sum()
    assert np.allclose(freq, np.minimum(p, q) + tv * p, atol=3e-3)


def test_exactness_does_not_depend_on_the_draft():
    """D-14.3 step 8.  A deliberately terrible q still emits exactly p; only
    the acceptance rate suffers."""
    p, _ = _pair()
    rng = np.random.default_rng(2)
    q = np.full(V, 1.0 / V)
    counts = np.bincount(speculative_emit(p, q, rng, N), minlength=V)
    assert chisquare(counts, p * N).pvalue > 0.01


@pytest.mark.parametrize("alpha,gamma", [(0.8, 4), (0.6, 8), (0.95, 16)])
def test_expected_tokens_matches_simulation(alpha, gamma):
    rng = np.random.default_rng(3)
    n = 400_000
    accepted = np.minimum(rng.geometric(1 - alpha, n) - 1, gamma)
    assert expected_tokens(alpha, gamma) == pytest.approx(accepted.mean() + 1,
                                                         abs=0.02)


def test_expected_tokens_limits():
    assert expected_tokens(0.8, 0) == pytest.approx(1.0)
    assert expected_tokens(1e-12, 4) == pytest.approx(1.0, abs=1e-9)
    assert expected_tokens(1 - 1e-12, 4) == pytest.approx(5.0, abs=1e-6)
