import numpy as np
import pytest
from exercises.ch08.solution import cross_entropy_from_logits


def _reference(logits, targets):
    z = logits - logits.max(-1, keepdims=True)
    lse = np.log(np.exp(z).sum(-1))
    return float((lse - z[np.arange(len(targets)), targets]).mean())


def test_logsumexp_matches_and_naive_overflows():
    """E-8.9.  The shift is exact, and it is not optional."""
    rng = np.random.default_rng(8)
    n, V = 64, 512
    logits = rng.normal(size=(n, V)).astype(np.float32)
    targets = rng.integers(0, V, size=n)
    assert abs(cross_entropy_from_logits(logits, targets)
               - _reference(logits, targets)) < 1e-6

    # at a logit scale of 100 the unshifted exponential is not representable in
    # float32, which is the precision a real model's logits arrive in
    big = (100.0 * logits).astype(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(big).sum(-1)
    assert np.isinf(naive).any(), "the naive route must actually overflow here"
    got = cross_entropy_from_logits(big, targets)
    assert np.isfinite(got), "yours must not"
    assert abs(got - _reference(big.astype(np.float64), targets)) < 1e-3


def test_shift_invariance():
    """D-8.2 step 3 as a test: adding a per-row constant changes nothing."""
    rng = np.random.default_rng(9)
    logits = rng.normal(size=(32, 128))
    targets = rng.integers(0, 128, size=32)
    shifted = logits + rng.normal(size=(32, 1)) * 50.0
    assert abs(cross_entropy_from_logits(logits, targets)
               - cross_entropy_from_logits(shifted, targets)) < 1e-9


def test_uniform_logits_give_log_V():
    """The anchor of D-8.2 step 6: a model that has learned nothing scores ln V."""
    V = 128256
    z = np.zeros((4, V))
    t = np.array([0, 1, 2, 3])
    assert abs(cross_entropy_from_logits(z, t) - np.log(V)) < 1e-9
