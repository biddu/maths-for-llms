"""E-11.13.  The decoupled logit is relative, and the compressed part is not.

D-11.3 splits each head into a part that absorbs and a part that carries
position.  The property the split has to preserve is the one RoPE existed to
give: the logit must depend on i and j only through i - j.  That is what these
tests check, and the second of them checks the failure mode as well, because
the failure is silent.
"""
import numpy as np

from exercises.ch11.solution import decoupled_rope_logits

D, D_C, D_H, D_R = 256, 128, 128, 64


def _weights(seed=14):
    rng = np.random.default_rng(seed)
    n_ = lambda *sh: rng.normal(size=sh) / np.sqrt(sh[0])
    return {"x_i": n_(D), "x_j": n_(D), "c_j": n_(D_C), "W_q": n_(D, D_H),
            "W_uk": n_(D_C, D_H), "W_qr": n_(D, D_R), "W_kr": n_(D, D_R)}


def _logit(w, i, j):
    return decoupled_rope_logits(w["x_i"], w["x_j"], w["c_j"], w["W_q"],
                                 w["W_uk"], w["W_qr"], w["W_kr"], i, j)


def test_relative_shift_invariance():
    """Shift both indices and the logit must not move.  Sixteen shifts, three
    starting pairs, because a construction can be accidentally invariant at one
    offset and not at others."""
    w = _weights()
    for i, j in ((20, 4), (33, 33), (40, 11)):
        base = _logit(w, i, j)
        for tau in range(1, 17):
            shifted = _logit(w, i + tau, j + tau)
            assert abs(shifted - base) < 1e-5, \
                "(%d,%d) shifted by %d moved by %.3e" % (i, j, tau,
                                                         abs(shifted - base))


def test_it_is_not_invariant_under_shifting_one_index():
    """The complementary check, and the one that shows the test above is not
    passing vacuously.  A logit that ignored position entirely would satisfy
    shift invariance perfectly, so the rotated term has to be contributing.

    The comparison is against the two-index shift rather than against an
    absolute tolerance: these logits are small, and what matters is that moving
    one index changes the answer by orders of magnitude more than moving both,
    which is a scale-free statement.
    """
    w = _weights()
    base = _logit(w, 20, 4)
    both = abs(_logit(w, 27, 11) - base)
    for i, j in ((21, 4), (20, 5), (24, 4)):
        one = abs(_logit(w, i, j) - base)
        assert one > 1e-6, "(%d,%d) barely moved: %.3e" % (i, j, one)
        assert one > 1e6 * max(both, 1e-15), (one, both)


def test_the_scale_is_over_the_full_width():
    """(11.10) is scaled by 1/sqrt(d_h + d_r), not 1/sqrt(d_h): the logit is a
    dot product in d_h + d_r dimensions and the variance argument of §3.3 is
    about the dimension actually summed over.  Using the wrong one biases every
    logit by a constant factor, which a softmax reads as a temperature change."""
    w = _weights()
    i, j = 20, 4
    got = _logit(w, i, j)
    content = (w["x_i"] @ (w["W_q"] @ w["W_uk"].T)) @ w["c_j"]

    def R(k, n, base=10000.0):
        th = base ** (-np.arange(0, n, 2) / n) * k
        M = np.zeros((n, n))
        for p, (c, s) in enumerate(zip(np.cos(th), np.sin(th))):
            M[2 * p:2 * p + 2, 2 * p:2 * p + 2] = [[c, s], [-s, c]]
        return M

    rot = ((w["x_i"] @ w["W_qr"]) @ R(i, D_R)) @ ((w["x_j"] @ w["W_kr"]) @ R(j, D_R))
    assert abs(got - (content + rot) / np.sqrt(D_H + D_R)) < 1e-8, \
        "scaled by sqrt(d_h) alone?  got %.6f, expected %.6f" % (
            got, (content + rot) / np.sqrt(D_H + D_R))
