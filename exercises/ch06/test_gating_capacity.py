import numpy as np
from exercises.ch06.solution import gated_basis, best_fit


def _target(d, seed=3):
    """A symmetric matrix with d/2 positive and d/2 negative eigenvalues."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    lam = np.array([3.0, 2.0, 1.5, 0.5, -0.5, -1.5, -2.0, -3.0])
    return Q @ np.diag(lam) @ Q.T, Q, lam


def test_factor_of_two():
    """E-6.10.  D-6.3 step 6: an ungated unit contributes a rank-1 Hessian and a
    gated unit a rank-2 one, so a gated layer of width n reaches what an ungated
    layer needs width 2n for.  No optimiser: this is a least-squares fit in three
    explicit spans, and the failure of the middle one is an Eckart-Young bound
    rather than a training artefact."""
    d = 8
    n = d // 2
    S, Q, _ = _target(d)
    assert np.allclose(S, S.T)

    gated_n = best_fit(gated_basis(Q, n), S)
    ungated_n = best_fit([np.outer(Q[:, j], Q[:, j]) for j in range(n)], S)
    ungated_2n = best_fit([np.outer(Q[:, j], Q[:, j]) for j in range(2 * n)], S)

    assert gated_n < 1e-10, "n gated units span a rank-2n Hessian"
    assert ungated_2n < 1e-10, "2n ungated units span a rank-2n Hessian"
    assert ungated_n > 0.3, "n ungated units cannot, and no choice of keys helps"


def test_eckart_young_bound_is_tight():
    """The best an ungated layer of width n can do over *all* keys is the best
    rank-n symmetric approximation.  Compute it and confirm the width-n fit
    above does not beat it."""
    d = 8
    n = d // 2
    S, Q, _ = _target(d)
    w, V = np.linalg.eigh(S)
    keep = np.argsort(-np.abs(w))[:n]
    Sn = sum(w[i] * np.outer(V[:, i], V[:, i]) for i in keep)
    bound = np.linalg.norm(Sn - S) / np.linalg.norm(S)
    got = best_fit([np.outer(Q[:, j], Q[:, j]) for j in range(n)], S)
    assert round(bound, 3) == 0.402
    assert got >= bound - 1e-12
