import numpy as np
import pytest
from exercises.ch07.solution import block_forward, block_backward

CFG = dict(d=32, h=4, n_kv=2, d_h=8, d_ff=88)
S = 12


def _init(seed=0):
    rng = np.random.default_rng(seed)
    c = CFG
    n = lambda *sh: rng.normal(size=sh) / np.sqrt(sh[0])
    return {"Q": n(c["d"], c["h"] * c["d_h"]), "K": n(c["d"], c["n_kv"] * c["d_h"]),
            "V": n(c["d"], c["n_kv"] * c["d_h"]), "O": n(c["h"] * c["d_h"], c["d"]),
            "gate": n(c["d"], c["d_ff"]), "up": n(c["d"], c["d_ff"]),
            "down": n(c["d_ff"], c["d"]),
            "g1": 1 + 0.05 * rng.normal(size=c["d"]),
            "g2": 1 + 0.05 * rng.normal(size=c["d"])}


def _worst_error(reduce="sum", n_probe=30, seed=1):
    """Central-difference check of every weight, scaled by that tensor's
    largest gradient entry.

    Scaling by the tensor's largest entry rather than by each entry's own value
    is deliberate, and E-7.10 asks you to say why: a plain per-entry relative
    error is dominated by entries whose true gradient is near zero, where the
    finite-difference floor is the entire signal, and it reports a failure that
    is not one.
    """
    rng = np.random.default_rng(seed)
    W = _init()
    x = rng.normal(size=(S, CFG["d"]))
    C = rng.normal(size=(S, CFG["d"]))
    loss = lambda: float((block_forward(x, W, CFG)[0] * C).sum())
    _, cache = block_forward(x, W, CFG)
    _, grads = block_backward(C, cache, W, CFG, reduce=reduce)
    worst = 0.0
    for k, A in W.items():
        flat, gflat = A.ravel(), grads[k].ravel()
        assert grads[k].shape == A.shape, "%s: gradient must have the weight's shape" % k
        scale = max(np.abs(gflat).max(), 1e-12)
        for i in rng.choice(flat.size, size=min(n_probe, flat.size), replace=False):
            o = flat[i]; h = 1e-5 * max(1.0, abs(o))
            flat[i] = o + h; lp = loss()
            flat[i] = o - h; lm = loss()
            flat[i] = o
            worst = max(worst, abs((lp - lm) / (2 * h) - gflat[i]) / scale)
    return worst


def test_block_backward_matches_central_difference():
    """E-7.10.  The signature exercise."""
    err = _worst_error("sum")
    assert err < 1e-7, "max scaled error %.2e" % err


def test_planted_group_mean_is_caught():
    """The same check must FAIL when equation (7.22)'s sum becomes a mean.

    A gradient check that passes on a known-wrong implementation is not a
    gradient check.  This test is the check on the check.
    """
    err = _worst_error("mean")
    assert err > 1e-3, ("averaging the group contributions must break the "
                        "gradient check; got %.2e" % err)


def test_gradient_wrt_input():
    """dL/dx as well as dL/dW: the block below receives it as its dL/dz."""
    rng = np.random.default_rng(2)
    W = _init()
    x = rng.normal(size=(S, CFG["d"]))
    C = rng.normal(size=(S, CFG["d"]))
    loss = lambda: float((block_forward(x, W, CFG)[0] * C).sum())
    _, cache = block_forward(x, W, CFG)
    xbar, _ = block_backward(C, cache, W, CFG)
    assert xbar.shape == x.shape
    flat, gflat = x.ravel(), xbar.ravel()
    scale = np.abs(gflat).max()
    worst = 0.0
    for i in rng.choice(flat.size, size=40, replace=False):
        o = flat[i]; h = 1e-5 * max(1.0, abs(o))
        flat[i] = o + h; lp = loss()
        flat[i] = o - h; lm = loss()
        flat[i] = o
        worst = max(worst, abs((lp - lm) / (2 * h) - gflat[i]) / scale)
    assert worst < 1e-7, "max scaled error %.2e" % worst
