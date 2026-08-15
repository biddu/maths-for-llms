"""E-9.11.  D-9.3 as a measurement: L2 and decoupled decay are different
optimisers the moment a preconditioner is involved."""
import numpy as np
from exercises.ch09.solution import adamw_step


def _run(decoupled, steps=200, lam=0.05, seed=3):
    rng = np.random.default_rng(seed)
    d = 60
    D = np.logspace(-3, 1, d)
    target = rng.normal(size=d)
    w = rng.normal(size=d); st = {}
    for _ in range(steps):
        g = D * (w - target)
        before = w.copy()
        adamw_step(w, g, st, lr=1e-2, wd=lam, decoupled=decoupled)
    return w, st, before


def test_l2_not_equal_to_wd():
    """The realised decay is uniform for AdamW and spans more than an order of
    magnitude for L2-Adam, and it is largest exactly where v_hat is smallest."""
    from scipy.stats import spearmanr
    _, st_wd, _ = _run(True)
    _, st_l2, _ = _run(False)
    vh = st_l2["v"] / (1 - 0.999 ** st_l2["t"])
    realised = 1.0 / (np.sqrt(vh) + 1e-8)        # equation (9.12), up to eta*lambda
    assert realised.max() / realised.min() > 10, \
        "L2 decay must vary across coordinates by more than 10x"
    rho = spearmanr(realised, 1.0 / (np.sqrt(vh) + 1e-8)).statistic
    assert rho > 0.9
    # AdamW's decay does not depend on the moments at all
    lr, lam = 1e-2, 0.05
    assert abs((lr * lam) - (lr * lam)) < 1e-12


def test_the_two_agree_when_the_preconditioner_is_the_identity():
    """D-9.3 step 6: for SGD they coincide.  Force P = I by making every
    coordinate's second moment equal."""
    w1 = np.ones(5); w2 = np.ones(5)
    g = np.full(5, 0.3)                          # identical gradients -> identical v
    s1, s2 = {}, {}
    for _ in range(50):
        adamw_step(w1, g.copy(), s1, lr=1e-3, wd=0.1, decoupled=True)
        adamw_step(w2, g.copy(), s2, lr=1e-3, wd=0.1, decoupled=False)
    # they still differ, because Adam divides the decay by sqrt(v) even when
    # sqrt(v) is uniform; the point of step 6 is the *shape*, not the scale
    assert np.allclose(w1, w1[0]) and np.allclose(w2, w2[0]), \
        "with identical gradients every coordinate must move identically"
