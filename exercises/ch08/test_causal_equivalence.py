"""E-8.11.  M-8.1 as a test: the mask removes a dependency, not a computation.

If the causal mask really makes position i depend only on positions <= i, then
running the whole sequence once and reading row i must give exactly what
running the prefix x[:i+1] gives in its last row.  That is the claim, and it is
what makes the s losses of equation (8.17) available from one pass.
"""
import time
import numpy as np
from exercises.ch08.solution import causal_block_forward, token_losses

S, D, V = 128, 24, 97


def _setup(seed=5):
    rng = np.random.default_rng(seed)
    n = lambda *sh: rng.normal(size=sh) / np.sqrt(sh[0])
    W = {k: n(D, D) for k in "QKVO"}
    X = rng.normal(size=(S, D))
    U = n(D, V)
    targets = rng.integers(0, V, size=S)
    return X, W, U, targets


def test_parallel_equals_sequential():
    X, W, U, targets = _setup()

    t0 = time.perf_counter()
    parallel = token_losses(X, W, U, targets)
    t_par = time.perf_counter() - t0
    assert parallel.shape == (S,)

    t0 = time.perf_counter()
    sequential = np.empty(S)
    for i in range(S):
        out = causal_block_forward(X[: i + 1], W)[-1]
        z = out @ U
        z = z - z.max()
        sequential[i] = float(np.log(np.exp(z).sum()) - z[targets[i]])
    t_seq = time.perf_counter() - t0

    err = np.abs(parallel - sequential).max()
    assert err < 1e-6, "max per-token difference %.2e" % err
    print("\n  s = %d: parallel %.4f s, sequential %.4f s, ratio %.1fx"
          % (S, t_par, t_seq, t_seq / max(t_par, 1e-9)))


def test_row_i_ignores_everything_after_it():
    """Perturb a later token and assert earlier rows do not move at all."""
    X, W, _, _ = _setup()
    full = causal_block_forward(X, W)
    Y = X.copy()
    Y[S // 2:] += 7.0
    moved = causal_block_forward(Y, W)
    assert np.abs(full[: S // 2] - moved[: S // 2]).max() < 1e-12
    assert np.abs(full[S // 2:] - moved[S // 2:]).max() > 1e-3
