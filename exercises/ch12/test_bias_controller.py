"""E-12.11.  The bias controller, its convergence, and its ripple.

The plant here is a fixed population of tokens routed through a fixed router,
so the only thing moving is the bias.  That isolates the controller from the
sampling noise a real training loop adds, which is deliberate: F-12.3 shows
what the noise does, and these tests show what the controller does.
"""
import numpy as np
import pytest

from exercises.ch12.solution import bias_controller

E, K, D, N = 32, 4, 64, 4096


def _plant(seed=31):
    """Returns loads_fn and the plant gain g_p measured on it."""
    rng = np.random.default_rng(seed)
    Wr = rng.normal(size=(D, E)) / np.sqrt(D)
    Z = rng.normal(size=(N, D)) @ Wr

    def loads_fn(gamma):
        sel = np.argpartition(-(Z + gamma), K, axis=1)[:, :K]
        return np.bincount(sel.ravel(), minlength=E) / (N * K)

    h = 0.02
    gains = []
    for i in range(E):
        e = np.zeros(E); e[i] = h
        gains.append((loads_fn(e)[i] - loads_fn(-e)[i]) / (2 * h))
    return loads_fn, float(np.mean(gains))


def test_bias_controller_converges():
    loads_fn, _ = _plant()
    start = np.abs(1.0 / E - loads_fn(np.zeros(E))).max()
    gamma, hist = bias_controller(loads_fn, E, K, u=0.01, steps=600)
    assert gamma.shape == (E,) and hist.shape == (600, E)
    final = np.abs(hist[-50:]).max(1).mean()
    assert final < start / 3, "start %.5f, finished at %.5f" % (start, final)
    # and the loads really are near uniform at the end
    c = loads_fn(gamma)
    assert c.max() / c.min() < 1.35, c.max() / c.min()


def test_the_ripple_scales_with_the_gain():
    """D-12.3 step 7.  A fixed-magnitude step cannot converge to a point: it
    limit-cycles, and the amplitude in load error is of order u * g_p.  This
    is the correction to the blueprint's u / g_p, which is not even
    dimensionally possible."""
    loads_fn, g_p = _plant()
    amps = {}
    for u in (0.01, 0.03, 0.1, 0.3):
        _, hist = bias_controller(loads_fn, E, K, u=u, steps=1200)
        amps[u] = np.abs(hist[-300:]).max(1).mean()
    assert amps[0.01] < amps[0.03] < amps[0.1] < amps[0.3], amps
    # once the ripple clears the quantisation floor of the load histogram, the
    # prediction holds to well within a factor of two
    for u in (0.03, 0.1, 0.3):
        assert 0.6 < amps[u] / (u * g_p) < 1.7, (u, amps[u], u * g_p)


def test_a_larger_gain_is_faster_and_worse():
    """The trade the gain buys, in one assertion each way."""
    loads_fn, _ = _plant()
    def steps_to(u, target):
        _, hist = bias_controller(loads_fn, E, K, u=u, steps=800)
        err = np.abs(hist).max(1)
        below = np.flatnonzero(err < target)
        return below[0] if len(below) else 10 ** 9
    # the target has to sit above the large gain's own ripple, or it never
    # arrives at all: that is the trade, not a failure of the test
    fast, slow = steps_to(0.1, 0.008), steps_to(0.003, 0.008)
    assert fast < slow, (fast, slow)
    _, h_fast = bias_controller(loads_fn, E, K, u=0.1, steps=1200)
    _, h_slow = bias_controller(loads_fn, E, K, u=0.003, steps=1200)
    assert np.abs(h_fast[-200:]).max(1).mean() > np.abs(h_slow[-200:]).max(1).mean()


def test_it_never_touches_the_gates():
    """Step 1 of D-12.3, as a shape argument.  The controller's only output is
    a bias on the *selection*; nothing it returns can enter the gating path,
    which is what makes its contribution to the objective identically zero."""
    loads_fn, _ = _plant()
    gamma, hist = bias_controller(loads_fn, E, K, u=0.01, steps=50)
    assert gamma.dtype.kind == "f" and gamma.shape == (E,)
    assert np.isfinite(gamma).all()
    # a constant shift of every bias changes nothing: only differences matter
    a = loads_fn(gamma)
    b = loads_fn(gamma + 3.7)
    assert np.allclose(a, b), "the controller's zero mode must be a no-op"


@pytest.mark.parametrize("u", [0.003, 0.03])
def test_history_records_the_error_not_the_load(u):
    loads_fn, _ = _plant()
    gamma, hist = bias_controller(loads_fn, E, K, u=u, steps=120)
    assert abs(hist.sum(1)).max() < 1e-9, \
        "each row must be 1/E - load, which sums to zero across experts"
