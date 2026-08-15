import numpy as np
from exercises.ch09.solution import adamw_step


def _reference(w, g, st, lr, b1, b2, eps, wd, decoupled):
    st.setdefault("m", np.zeros_like(w)); st.setdefault("v", np.zeros_like(w))
    st["t"] = st.get("t", 0) + 1
    gg = g if decoupled else g + wd * w
    st["m"] = b1 * st["m"] + (1 - b1) * gg
    st["v"] = b2 * st["v"] + (1 - b2) * gg ** 2
    mh = st["m"] / (1 - b1 ** st["t"]); vh = st["v"] / (1 - b2 ** st["t"])
    w -= lr * mh / (np.sqrt(vh) + eps)
    if decoupled:
        w -= lr * wd * w
    return w


def test_adamw_matches_reference():
    """E-9.10.  100 steps on an ill-conditioned quadratic."""
    rng = np.random.default_rng(9)
    d = 40
    D = np.logspace(-3, 1, d)                    # condition number 10^4
    target = rng.normal(size=d)
    w1 = rng.normal(size=d); w2 = w1.copy()
    s1, s2 = {}, {}
    for _ in range(100):
        g1 = D * (w1 - target)
        g2 = D * (w2 - target)
        adamw_step(w1, g1, s1, lr=1e-2, wd=0.01, decoupled=True)
        _reference(w2, g2, s2, 1e-2, 0.9, 0.999, 1e-8, 0.01, True)
    assert np.abs(w1 - w2).max() / max(np.abs(w2).max(), 1e-12) < 1e-6


def test_first_step_is_a_sign_step():
    """D-9.2 step 8: at t = 1 the bias corrections cancel and the update is
    exactly -lr * sign(g), whatever the gradient's magnitude."""
    for scale in (1e-6, 1.0, 1e6):
        w = np.zeros(8); g = scale * np.array([1., -2., 3., -4., 5., -6., 7., -8.])
        adamw_step(w, g, {}, lr=0.1, eps=0.0)
        assert np.allclose(w, -0.1 * np.sign(g)), "scale %g" % scale
