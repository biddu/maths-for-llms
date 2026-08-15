"""E-15.11.  The reward-model gradient (15.5), against a numerical derivative.

The factor that matters is sigma(-Delta).  An implementation that drops it
still descends, because the direction is right whenever the pair is backwards;
it is wrong on every pair the model already has right, which is most of them
after a few hundred steps.  A finite difference notices.  A loss curve does not.
"""
import numpy as np
import pytest

from exercises.ch15.solution import rm_loss_and_grad

D_IN, H, N = 12, 16, 64


def _setup(seed=1511):
    rng = np.random.default_rng(seed)
    params = [rng.standard_normal((D_IN, H)) / np.sqrt(D_IN),
              rng.standard_normal((H, H)) / np.sqrt(H),
              rng.standard_normal(H) / np.sqrt(H)]
    return params, rng.standard_normal((N, D_IN)), rng.standard_normal((N, D_IN))


def test_analytic_matches_numeric():
    params, Xw, Xl = _setup()
    L, G = rm_loss_and_grad(params, Xw, Xl)
    assert np.isfinite(L) and L > 0
    assert len(G) == 3
    rng = np.random.default_rng(0)
    eps = 1e-6
    for k in range(3):
        flat = params[k].ravel()
        for i in rng.choice(flat.size, 8, replace=False):
            old = flat[i]
            flat[i] = old + eps
            Lp, _ = rm_loss_and_grad(params, Xw, Xl)
            flat[i] = old - eps
            Lm, _ = rm_loss_and_grad(params, Xw, Xl)
            flat[i] = old
            assert G[k].ravel()[i] == pytest.approx((Lp - Lm) / (2 * eps),
                                                    abs=1e-6)


def test_swapping_the_pair_negates_the_margin():
    params, Xw, Xl = _setup()
    L1, _ = rm_loss_and_grad(params, Xw, Xl)
    L2, _ = rm_loss_and_grad(params, Xl, Xw)
    # -log sigma(D) + -log sigma(-D) = -log(sigma(D) sigma(-D)) >= 2 log 2
    assert L1 + L2 >= 2 * np.log(2) - 1e-9


def test_a_confident_correct_pair_contributes_almost_nothing():
    """Self-annealing, D-15.2 step 6: the gain is sigma(-Delta).

    Compared at FIXED parameters, so the only thing that changes between the
    two calls is the margin.  Scaling the parameters instead would scale the
    gradient of r as well and measure nothing.
    """
    params, Xw, Xl = _setup()
    # a direction that raises the score, found by a one-sided difference
    eps = 1e-4
    base, _ = rm_loss_and_grad(params, Xw, Xl)
    u = np.zeros(D_IN)
    for j in range(D_IN):
        d = np.zeros(D_IN)
        d[j] = eps
        up, _ = rm_loss_and_grad(params, Xw + d, Xl)
        u[j] = (base - up) / eps            # descending the loss raises s(Xw)
    u = u / np.linalg.norm(u)
    _, g_right = rm_loss_and_grad(params, Xw + 6.0 * u, Xl - 6.0 * u)
    _, g_wrong = rm_loss_and_grad(params, Xw - 6.0 * u, Xl + 6.0 * u)
    n_right = sum(float(np.abs(g).sum()) for g in g_right)
    n_wrong = sum(float(np.abs(g).sum()) for g in g_wrong)
    assert n_right < 0.25 * n_wrong
