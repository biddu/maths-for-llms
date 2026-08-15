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
    SmolLM2-135M: raw 0.245, mean-centred 0.124, mean + top-1 removed 0.003."""
    rng = np.random.default_rng(5)
    mu = rng.normal(size=64); mu /= np.linalg.norm(mu)
    top = rng.normal(size=64); top /= np.linalg.norm(top)
    W = 5.0 * mu + 4.0 * np.outer(rng.normal(size=200), top) + rng.normal(size=(200, 64))
    off = ~np.eye(200, dtype=bool)
    cen = centred_cosine(W)
    abt = all_but_the_top(W, k=1)
    assert abs(abt[off].mean()) < abs(cen[off].mean()) / 3, \
        "removing the top direction must do most of the remaining work"
    assert abs(abt[off].mean()) < 0.05
