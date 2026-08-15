import numpy as np
from exercises.ch07.solution import attention_backward

CFG = dict(h=4, n_kv=2, d_h=8, s=12)


def _forward(Q, K, V, group):
    """The forward pass of equation (7.15), which Chapter 3 already gives you.
    Returned so the test can finite-difference through it; the exercise is the
    backward."""
    h, s, d_h = Q.shape
    mask = np.triu(np.full((s, s), -np.inf), 1)
    P = np.empty((h, s, s)); O = np.empty((h, s, d_h))
    for i in range(h):
        kv = i // group
        S = Q[i] @ K[kv].T / np.sqrt(d_h) + mask
        S = S - S.max(-1, keepdims=True)
        e = np.exp(S); P[i] = e / e.sum(-1, keepdims=True)
        O[i] = P[i] @ V[kv]
    return O, P


def _tensors(seed=3):
    c = CFG; rng = np.random.default_rng(seed)
    g = c["h"] // c["n_kv"]
    Q = rng.normal(size=(c["h"], c["s"], c["d_h"]))
    K = rng.normal(size=(c["n_kv"], c["s"], c["d_h"]))
    V = rng.normal(size=(c["n_kv"], c["s"], c["d_h"]))
    C = rng.normal(size=(c["h"], c["s"], c["d_h"]))
    return Q, K, V, C, g


def test_matches_finite_difference():
    """E-7.9.  Central differences through the forward pass, in float64."""
    Q, K, V, C, g = _tensors()
    O, P = _forward(Q, K, V, g)
    dQ, dK, dV = attention_backward(C, Q, K, V, P, g)
    assert dQ.shape == Q.shape and dK.shape == K.shape and dV.shape == V.shape

    loss = lambda: float((_forward(Q, K, V, g)[0] * C).sum())
    rng = np.random.default_rng(4)
    worst = 0.0
    for T, D in ((Q, dQ), (K, dK), (V, dV)):
        flat, gflat = T.ravel(), D.ravel()
        scale = np.abs(gflat).max()
        for i in rng.choice(flat.size, size=40, replace=False):
            o = flat[i]; h = 1e-5 * max(1.0, abs(o))
            flat[i] = o + h; lp = loss()
            flat[i] = o - h; lm = loss()
            flat[i] = o
            worst = max(worst, abs((lp - lm) / (2 * h) - gflat[i]) / scale)
    assert worst < 1e-7, "max error %.2e, scaled by the tensor's largest entry" % worst


def test_group_contributions_add_rather_than_average():
    """D-7.3 step 9.  A forward broadcast is a backward sum, and the difference
    is exactly the group size."""
    Q, K, V, C, g = _tensors()
    _, P = _forward(Q, K, V, g)
    _, dK_sum, dV_sum = attention_backward(C, Q, K, V, P, g, reduce="sum")
    _, dK_mean, dV_mean = attention_backward(C, Q, K, V, P, g, reduce="mean")
    assert np.allclose(dK_sum, g * dK_mean)
    assert np.allclose(dV_sum, g * dV_mean)
    assert not np.allclose(dK_sum, dK_mean), "g = %d, so these must differ" % g
