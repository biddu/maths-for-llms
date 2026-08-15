import numpy as np
from exercises.ch07.solution import softmax_backward


def test_matches_explicit_jacobian():
    """E-7.8.  Equation (7.6) against J = diag(p) - p p^T, built explicitly.

    The explicit route is what the exercise exists to avoid, so it appears
    here once, at s = 256, where it still fits.
    """
    rng = np.random.default_rng(7)
    s = 256
    z = rng.normal(size=s)
    p = np.exp(z - z.max()); p /= p.sum()
    g = rng.normal(size=s)
    J = np.diag(p) - np.outer(p, p)
    assert np.allclose(J, J.T), "the softmax Jacobian is symmetric (D-7.2 step 2)"
    assert np.abs(softmax_backward(p, g) - J @ g).max() < 1e-12


def test_batched_rows():
    """A whole attention matrix at once, equation (7.7)."""
    rng = np.random.default_rng(8)
    Z = rng.normal(size=(4, 32, 32))
    P = np.exp(Z - Z.max(-1, keepdims=True))
    P /= P.sum(-1, keepdims=True)
    G = rng.normal(size=P.shape)
    got = softmax_backward(P, G)
    assert got.shape == P.shape
    for i in range(P.shape[0]):
        for t in range(P.shape[1]):
            p, g = P[i, t], G[i, t]
            want = (np.diag(p) - np.outer(p, p)) @ g
            assert np.abs(got[i, t] - want).max() < 1e-12


def test_only_contrast_survives():
    """D-7.2 step 7: adding a constant to g changes nothing, because the
    logits are defined only up to a per-row additive constant."""
    rng = np.random.default_rng(9)
    z = rng.normal(size=64)
    p = np.exp(z - z.max()); p /= p.sum()
    g = rng.normal(size=64)
    assert np.abs(softmax_backward(p, g) - softmax_backward(p, g + 3.7)).max() < 1e-13


def test_masked_entries_stay_zero():
    """A -inf mask gives p = 0 exactly, so the backward is zero there without
    any re-masking.  Masking with -1e9 does not, which is the failure mode."""
    rng = np.random.default_rng(10)
    s = 32
    z = rng.normal(size=(s, s)) + np.triu(np.full((s, s), -np.inf), 1)
    P = np.exp(z - z.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
    G = rng.normal(size=(s, s))
    out = softmax_backward(P, G)
    assert np.all(np.triu(out, 1) == 0.0)
