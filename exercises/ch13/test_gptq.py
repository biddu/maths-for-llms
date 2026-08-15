"""E-13.12.  GPTQ beats round-to-nearest, and the size of the win is a fact
about the calibration data rather than about the method.

D-13.3's compensation pushes error along H^-1's j-th column.  If the
calibration activations are uncorrelated then H is nearly diagonal, that column
is nearly zero away from j, and there is nowhere to push.  Both cases are
tested, because the contrast is the lesson: calibrating off distribution does
not merely make GPTQ suboptimal, it can remove its reason to exist.
"""
import numpy as np

from exercises.ch13.solution import gptq

N, K, M = 512, 512, 2048


def _round_to_nearest(W, b_q=4, g_q=128):
    Q = np.empty_like(W)
    for s in range(0, W.shape[0], g_q):
        blk = W[s:s + g_q]
        lo, hi = blk.min(0), blk.max(0)
        sc = np.maximum((hi - lo) / (2 ** b_q - 1), 1e-12)
        z = np.round(-lo / sc)
        Q[s:s + g_q] = sc * (np.clip(np.round(blk / sc) + z, 0, 2 ** b_q - 1) - z)
    return Q


def _calibration(kind, seed=7):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((M, N))
    if kind == "independent":
        X = Z
    else:
        # a power-law covariance spectrum, which is what real activations have
        U, _ = np.linalg.qr(rng.standard_normal((N, N)))
        X = (Z * (np.arange(1, N + 1) ** -0.5)) @ U.T
    X[:, rng.choice(N, 15, replace=False)] *= 12.0
    return X


def _weights(seed=41):
    return np.random.default_rng(seed).standard_normal((N, K)) / np.sqrt(N)


def _gain(kind, act_order=False):
    X, W = _calibration(kind), _weights()
    H = 2 * X.T @ X
    e_rtn = np.linalg.norm(X @ W - X @ _round_to_nearest(W))
    e_gptq = np.linalg.norm(X @ W - X @ gptq(W, H, act_order=act_order))
    return 1.0 - e_gptq / e_rtn


def test_gptq_beats_rtn():
    """On correlated calibration data, which is the case that matters."""
    gain = _gain("correlated")
    assert gain >= 0.30, "only %.1f%% better than round-to-nearest" % (100 * gain)


def test_and_barely_beats_it_on_independent_data():
    """The complementary measurement, and the one that explains the first."""
    corr, indep = _gain("correlated"), _gain("independent")
    assert indep < 0.15, "%.1f%% on independent rows" % (100 * indep)
    assert corr > 2 * indep, (corr, indep)


def test_ordering_helps():
    """Visiting high-Hessian-diagonal coordinates first means the coordinates
    that matter most are quantised while the most compensation budget is still
    unspent."""
    assert _gain("correlated", act_order=True) > _gain("correlated") 


def test_it_returns_something_on_the_grid():
    """A compensated weight is still a quantised weight: the sweep changes
    which grid point each coordinate lands on, never whether it is on one."""
    X, W = _calibration("correlated"), _weights()
    Q = gptq(W, 2 * X.T @ X)
    assert Q.shape == W.shape
    for s in range(0, N, 128):
        blk = Q[s:s + 128]
        for col in range(0, K, 97):
            vals = np.unique(np.round(blk[:, col], 10))
            assert len(vals) <= 16, (s, col, len(vals))
