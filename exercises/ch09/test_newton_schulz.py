"""E-9.12.  What the iteration guarantees, and what it does not.

The blueprint asked for a relative error below 0.1 against U V^T.  That is not
attainable and the chapter says why: the polynomial has fixed points at 0.868
and 1.264 and oscillates between them rather than converging to 1.  What it
does guarantee is exact preservation of the singular vectors and a bounded
band of singular values, and those are what the oracle of (9.26) needs.
"""
import numpy as np
from exercises.ch09.solution import newton_schulz


def test_orthogonalises():
    rng = np.random.default_rng(12)
    G = rng.normal(size=(4096, 1024))
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    X = newton_schulz(G, steps=5)
    assert X.shape == G.shape

    # the singular values land in a band around 1
    sx = np.linalg.svd(X, compute_uv=False)
    assert 0.6 <= sx.min() and sx.max() <= 1.25, "band was [%.3f, %.3f]" % (sx.min(), sx.max())

    # the direction is right, to the tolerance the polynomial permits
    ref = U @ Vt
    rel = np.linalg.norm(X - ref) / np.linalg.norm(ref)
    assert rel < 0.25, "relative error %.3f" % rel


def test_singular_vectors_are_untouched():
    """D-9.4 step 5.  Not approximately preserved: preserved."""
    rng = np.random.default_rng(13)
    G = rng.normal(size=(64, 200))
    U, _, Vt = np.linalg.svd(G, full_matrices=False)
    X = newton_schulz(G, steps=5)
    PU, PV = U @ U.T, Vt.T @ Vt
    assert np.linalg.norm(X - PU @ X) / np.linalg.norm(X) < 1e-10
    assert np.linalg.norm(X - X @ PV) / np.linalg.norm(X) < 1e-10


def test_more_iterations_do_not_help():
    """The iteration is not converging, so seven steps are not better than five.
    E-9.12 asks the reader to explain this; the test just pins it."""
    rng = np.random.default_rng(14)
    G = rng.normal(size=(512, 256))
    U, _, Vt = np.linalg.svd(G, full_matrices=False)
    ref = U @ Vt
    err = [np.linalg.norm(newton_schulz(G, steps=k) - ref) / np.linalg.norm(ref)
           for k in (5, 7, 9)]
    assert min(err) > 0.05, "it does not converge to U V^T"
    assert err[1] >= err[0] - 0.02, "and seven steps are no better than five"
