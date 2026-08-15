"""E-12.9.  The router, and the property that makes it trainable.

Renormalising over the selected set is not cosmetic.  It is what makes the
gradient with respect to an unselected logit exactly zero (D-12.1), and the
last test here is that derivation checked numerically rather than trusted.
"""
import numpy as np
import pytest

from exercises.ch12.solution import route

T, E, K = 64, 16, 4


def _logits(seed=12, scale=1.5):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(T, E)) * scale


def test_router_topk_renormalise():
    z = _logits()
    idx, ghat = route(z, K)
    assert idx.shape == (T, K) and ghat.shape == (T, K)
    # exactly k distinct experts per token
    for t in range(T):
        assert len(set(idx[t].tolist())) == K, idx[t]
    # the selected gates form a distribution
    assert np.allclose(ghat.sum(1), 1.0, atol=1e-10)
    assert (ghat > 0).all()


def test_it_selects_the_largest_logits():
    """Selection is on z, not on the gate.  Softmax is monotone so the two
    agree here, but an implementation that sorts the wrong array will fail the
    moment a bias is added in E-12.11."""
    z = _logits(seed=3)
    idx, _ = route(z, K)
    for t in range(T):
        chosen = set(idx[t].tolist())
        rest = [i for i in range(E) if i not in chosen]
        assert z[t, list(chosen)].min() >= z[t, rest].max()


def test_the_gates_are_the_softmax_restricted_and_rescaled():
    z = _logits(seed=4)
    idx, ghat = route(z, K)
    g = np.exp(z - z.max(1, keepdims=True))
    g /= g.sum(1, keepdims=True)
    sel = np.take_along_axis(g, idx, 1)
    assert np.allclose(ghat, sel / sel.sum(1, keepdims=True), atol=1e-12), \
        "renormalised over all E rather than over the selected set?"


@pytest.mark.parametrize("k", [1, 2, 8])
def test_it_works_at_other_k(k):
    z = _logits(seed=5)
    idx, ghat = route(z, k)
    assert idx.shape == (T, k)
    assert np.allclose(ghat.sum(1), 1.0, atol=1e-10)
    if k == 1:
        assert np.allclose(ghat, 1.0), "at k = 1 the single gate must be 1"


def test_the_unselected_gradient_is_zero():
    """D-12.1, measured.  Perturb an unselected logit and the layer's output
    does not move; perturb a selected one and it does.  The comparison is the
    point: without renormalisation the first number is not zero either.
    """
    rng = np.random.default_rng(7)
    z = rng.normal(size=E) * 1.5
    experts = rng.normal(size=(E, 8))          # E_i(x), constant in z

    def out(zz, renorm=True):
        idx, ghat = route(zz[None, :], K)
        if not renorm:                          # the counterfactual of step 7
            g = np.exp(zz - zz.max()); g /= g.sum()
            ghat = g[idx]
        return (ghat[0][:, None] * experts[idx[0]]).sum(0)

    idx, _ = route(z[None, :], K)
    chosen = set(idx[0].tolist())
    unchosen = [i for i in range(E) if i not in chosen]
    h = 1e-6

    def slope(m, renorm):
        up, dn = z.copy(), z.copy()
        up[m] += h; dn[m] -= h
        return np.abs(out(up, renorm) - out(dn, renorm)).max() / (2 * h)

    assert max(slope(m, True) for m in unchosen) < 1e-6
    assert max(slope(m, True) for m in chosen) > 1e-3
    assert max(slope(m, False) for m in unchosen) > 1e-3, \
        "without renormalisation the unselected gradient must survive"
