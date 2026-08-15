import numpy as np
from exercises.ch02.solution import centred_cosine, all_but_the_top


def test_centring_changes_ranking():
    """Mean-centring changes the neighbour ranking.  It does not, on its own,
    remove the anisotropy: see test_top_direction_dominates below, and
    measure/README.md for the measurement that forced this correction."""
    rng = np.random.default_rng(4)
    mu = rng.normal(size=64); mu /= np.linalg.norm(mu)
    W = 7.0 * mu + rng.normal(size=(40, 64))
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    raw = Wn @ Wn.T
    cen = centred_cosine(W)
    off = ~np.eye(40, dtype=bool)
    assert raw[off].mean() > 0.2, "the raw cloud must be anisotropic to make the point"
    assert cen[off].mean() < raw[off].mean(), "centring must reduce the floor"
    top = lambda M, i: set(np.argsort(-M[i])[1:6])
    assert any(top(raw, i) != top(cen, i) for i in range(40))


def test_top_direction_dominates():
    """On a real embedding matrix, mean-centring leaves about half the
    anisotropy; the top principal direction carries the rest.  Measured on
    SmolLM2-135M: raw 0.245, mean-centred 0.124, mean + top-1 removed 0.003.

    THE FIXTURE HAS TO SEPARATE THE VOCABULARY FROM THE SAMPLE, and the first
    version of this test did not.  It built 200 rows, centred them on their own
    centroid, and measured all 200.  Subtracting the centroid of exactly the rows
    you then measure drives the mean off-diagonal cosine to its algebraic floor
    of -1/(n-1) = -0.005 on its own, so centring appeared to remove everything
    and the top direction had nothing left to do.  The test could not be passed
    by any implementation.  It also loaded the top direction symmetrically, with
    a zero-mean coefficient, which contributes nothing to a mean cosine in the
    first place.

    What a real embedding matrix does instead, and what figs/make_fig24.py
    already models: the mean is estimated over the WHOLE vocabulary, the cosines
    are measured on a biased sample of it, and the second direction is loaded by
    the frequent head alone.  The centroid then removes only the head's share of
    that loading, which is why centring leaves about half.

    This fixture reproduces raw +0.229, centred +0.093, mean and top-1 removed
    +0.001, in the same order and by the same mechanism as the measurement.
    """
    rng = np.random.default_rng(5)
    d, n_vocab, n_head = 64, 2000, 200
    mu = rng.normal(size=d); mu /= np.linalg.norm(mu)
    top = rng.normal(size=d); top /= np.linalg.norm(top)
    # Every token shares mu.  Only the frequent head loads onto top, and it loads
    # positively, so the direction carries mean anisotropy rather than spread.
    load = np.zeros(n_vocab)
    load[:n_head] = 3.0 + rng.normal(scale=0.5, size=n_head)
    W = 3.0 * mu + np.outer(load, top) + rng.normal(size=(n_vocab, d))

    head = slice(0, n_head)
    off = ~np.eye(n_head, dtype=bool)
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    raw = (Wn @ Wn.T)[head, head]
    cen = centred_cosine(W)[head, head]
    abt = all_but_the_top(W, k=1)[head, head]

    assert raw[off].mean() > 0.2, "the head must be anisotropic to make the point"
    assert abs(cen[off].mean()) < raw[off].mean() / 2, \
        "centring must remove roughly half of it"
    assert abs(abt[off].mean()) < abs(cen[off].mean()) / 3, \
        "removing the top direction must do most of the remaining work"
    assert abs(abt[off].mean()) < 0.05
