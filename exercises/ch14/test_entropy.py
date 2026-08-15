"""E-14.3.  Entropy is monotone in temperature, and both limits are what
D-14.1 step 7 says they are once the genericity hypothesis is put back.

The blueprint's step 7 sends H to zero as T -> 0+.  That needs a unique argmax.
With an m-fold tie the limit is log m, which is what this file asserts, and it
is the one place in the chapter where a stated limit had a missing hypothesis.
"""
import numpy as np
import pytest

from exercises.ch14.solution import dentropy_dT, entropy, temper

TS = np.geomspace(0.05, 20.0, 120)


def _logits(V=256, scale=2.0, seed=1403):
    return np.random.default_rng(seed).standard_normal(V) * scale


def test_entropy_monotone_and_limits():
    z = _logits()
    V = len(z)
    H = np.array([entropy(z, float(T)) for T in TS])
    assert np.all(np.diff(H) > -1e-12), "H must be non-decreasing in T"
    # (14.7): the gap closes like sigma_z^2 / 2T^2, so a fixed absolute
    # tolerance at T = 20 is a statement about the logit spread and not about
    # the implementation.  Assert the rate instead.
    gap = np.log(V) - H[-1]
    assert gap == pytest.approx(z.var() / (2 * TS[-1] ** 2), rel=0.01)
    # and the lower limit is log m with m the size of the argmax set, which is
    # 1 here, so H -> 0; at T = 0.05 it need not be small, because how fast it
    # falls depends on the gap between the two largest logits
    assert entropy(z, 1e-3) < 1e-9


def test_tied_argmax_tends_to_log_m():
    for m in (2, 3, 5):
        z = np.concatenate([np.full(m, 4.0), np.linspace(1.0, -3.0, 20)])
        assert entropy(z, 1e-2) == pytest.approx(np.log(m), abs=1e-6)


@pytest.mark.parametrize("T", [0.3, 0.7, 1.0, 1.8, 3.0])
def test_derivative_matches_central_difference(T):
    """The exponent is three.  A T**2 implementation passes at T = 1 alone,
    which is why this test is parameterised away from it."""
    z = _logits(V=64)
    h = 1e-6
    cd = (entropy(z, T + h) - entropy(z, T - h)) / (2 * h)
    assert dentropy_dT(z, T) == pytest.approx(cd, rel=1e-5)


def test_gap_to_log_V_closes_like_T_squared():
    """(14.7), obtained by integrating (14.4) from T to infinity."""
    z = _logits(V=1024, scale=3.0, seed=7)
    for T in (10.0, 20.0, 50.0):
        gap = np.log(len(z)) - entropy(z, T)
        assert gap == pytest.approx(z.var() / (2 * T ** 2), rel=0.01)


def test_temper_preserves_support_and_order():
    z = _logits(V=512)
    order = np.argsort(-z)
    for T in TS:
        p = temper(z, float(T))
        assert p.min() > 0.0
        assert np.array_equal(np.argsort(-p), order)
        assert p.sum() == pytest.approx(1.0, rel=1e-12)
