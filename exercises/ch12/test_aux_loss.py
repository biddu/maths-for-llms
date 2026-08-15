"""E-12.10.  The auxiliary loss, its gradient, and the identity behind it.

Three separate claims and they fail in different ways.  The value is an inner
product; the identity (12.5) turns that inner product into a variance and only
holds at P = f; and the gradient (12.7) is where the mechanism lives, because
it is the line that says *which* logit gets pushed down and when.
"""
import numpy as np

from exercises.ch12.solution import route, aux_loss

T, E, K = 512, 32, 4
ALPHA = 0.01


def _setup(seed=21, scale=1.5):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(T, E)) * scale
    idx, _ = route(z, K)
    return z, idx


def _f(idx):
    return np.bincount(idx.ravel(), minlength=E) / idx.size


def test_aux_loss_gradient():
    z, idx = _setup()
    loss, grad = aux_loss(z, idx, ALPHA)
    assert grad.shape == z.shape
    assert np.isfinite(loss) and loss > 0

    # against central differences at a scattered set of entries
    h = 1e-6
    worst = 0.0
    rng = np.random.default_rng(1)
    for t, j in zip(rng.integers(0, T, 24), rng.integers(0, E, 24)):
        up, dn = z.copy(), z.copy()
        up[t, j] += h; dn[t, j] -= h
        # the selection is held fixed: f is a count and does not move with z
        num = (aux_loss(up, idx, ALPHA)[0] - aux_loss(dn, idx, ALPHA)[0]) / (2 * h)
        worst = max(worst, abs(num - grad[t, j]))
    assert worst < 1e-6, "max deviation from finite differences: %.3e" % worst


def test_the_load_carries_no_gradient():
    """f is a count of assignments, so it is piecewise constant in z.  An
    implementation that differentiates through it will pass nothing here, and
    will also be minimising a different function from (12.6)."""
    z, idx = _setup()
    _, grad = aux_loss(z, idx, ALPHA)
    g = np.exp(z - z.max(1, keepdims=True)); g /= g.sum(1, keepdims=True)
    f = _f(idx)
    analytic = (ALPHA * E / T) * g * (f[None, :] - (g * f[None, :]).sum(1, keepdims=True))
    assert np.abs(grad - analytic).max() < 1e-12


def test_the_variance_identity():
    """(12.5).  Only at P = f, which is the assumption D-12.2 flags."""
    z, idx = _setup()
    f = _f(idx)
    direct = ALPHA * E * float(f @ f)
    identity = ALPHA + ALPHA * E ** 2 * float(np.var(f))
    assert abs(direct - identity) < 1e-12
    # uniform load attains the minimum, and the minimum does not depend on E
    uniform = np.full(E, 1.0 / E)
    assert abs(ALPHA * E * float(uniform @ uniform) - ALPHA) < 1e-15
    for e in (8, 64, 256):
        u = np.full(e, 1.0 / e)
        assert abs(ALPHA * e * float(u @ u) - ALPHA) < 1e-15


def test_the_sign_rule():
    """The mechanism, stated as E-12.1 asks: expert j's logit is pushed down
    exactly when its load exceeds this token's probability-weighted mean load.
    Note the comparison is against <f, g_t> and not against 1/E, so the same
    expert can be pushed down for one token and up for another."""
    z, idx = _setup()
    _, grad = aux_loss(z, idx, ALPHA)
    g = np.exp(z - z.max(1, keepdims=True)); g /= g.sum(1, keepdims=True)
    f = _f(idx)
    weighted_mean = (g * f[None, :]).sum(1)
    pushed_down = grad > 0                       # descent subtracts the gradient
    overloaded = f[None, :] > weighted_mean[:, None]
    assert (pushed_down == overloaded).mean() > 0.999
    # and the same expert really does go both ways across tokens
    both = ((pushed_down.any(0)) & (~pushed_down).any(0)).sum()
    assert both > E // 2, "the rule is per token, not per expert"
