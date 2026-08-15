"""E-14.11.  The four sampler operations, and the two structures they act on.

M-14.1's claim in executable form: temperature moves the log-odds and keeps the
support; truncation moves the support and keeps the odds among survivors.  A
correct implementation satisfies both halves, and an implementation that
renormalises in the wrong place satisfies neither.
"""
import numpy as np
import pytest

from exercises.ch14.solution import compose, min_p, temper, top_k, top_p


def _p(V=2048, scale=2.5, seed=1411):
    return temper(np.random.default_rng(seed).standard_normal(V) * scale, 1.0)


def test_temperature_preserves_support_and_order():
    rng = np.random.default_rng(7)
    z = rng.standard_normal(1024) * 3.0
    order = np.argsort(-z)
    for T in np.geomspace(0.05, 20.0, 60):
        p = temper(z, float(T))
        assert np.count_nonzero(p) == len(z)
        assert np.array_equal(np.argsort(-p), order)


@pytest.mark.parametrize("rule,arg", [(top_k, 40), (top_p, 0.9), (min_p, 0.05)])
def test_truncation_produces_exact_zeros(rule, arg):
    p = _p()
    out = rule(p, arg)
    assert np.count_nonzero(out) < len(p)
    assert np.all(out[out == 0] == 0.0)          # exact, not merely small
    assert out.sum() == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("rule,arg", [(top_k, 40), (top_p, 0.9), (min_p, 0.05)])
def test_truncation_preserves_odds_among_survivors(rule, arg):
    p = _p()
    out = rule(p, arg)
    keep = out > 0
    ratio = out[keep] / p[keep]
    assert np.allclose(ratio, ratio[0], rtol=1e-12)


def test_nucleus_is_minimum_cardinality():
    """(14.8) asks for the SMALLEST set reaching the threshold, so removing any
    kept coordinate must drop the retained mass below it."""
    p = _p()
    out = top_p(p, 0.9)
    kept = p[out > 0]
    assert kept.sum() >= 0.9
    assert kept.sum() - kept.min() < 0.9


def test_min_p_is_a_logit_window():
    """(14.9): p_i >= tau p_max iff z_max - z_i <= T log(1/tau)."""
    rng = np.random.default_rng(3)
    z = rng.standard_normal(4096) * 2.0
    for T in (0.5, 1.0, 2.0):
        kept = set(np.flatnonzero(min_p(temper(z, T), 0.05) > 0))
        window = set(np.flatnonzero(z.max() - z <= T * np.log(1 / 0.05)))
        assert kept == window


def test_nucleus_size_is_monotone_in_temperature():
    """Both rules widen monotonically, by (14.10).  The widely repeated claim
    that top-p's does not is false; see the chapter's author note."""
    rng = np.random.default_rng(5)
    z = rng.standard_normal(2048) * 2.0
    sizes = [np.count_nonzero(top_p(temper(z, float(T)), 0.9))
             for T in np.geomspace(0.2, 5.0, 40)]
    assert all(b >= a for a, b in zip(sizes, sizes[1:]))


def test_composition_order_matters():
    rng = np.random.default_rng(11)
    z = rng.standard_normal(4096) * 2.0
    a = compose(z, 2.0, "top_p", 0.9, temperature_first=True)
    b = compose(z, 2.0, "top_p", 0.9, temperature_first=False)
    assert np.count_nonzero(a) != np.count_nonzero(b)
    assert 0.5 * np.abs(a - b).sum() > 0.1
