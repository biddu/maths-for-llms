"""Your solutions for Chapter 9's [C] exercises.

THIS IS THE `solutions` BRANCH.  Every function below is worked in full and
every test in this directory passes.  On `main` these are stubs raising
NotImplementedError, and making them pass is the exercise; read this file only
after you have had a go, or when you want to compare a method rather than an
answer.  Appendix C prints the answer and the tolerance, not the code.
"""
import numpy as np


def adamw_step(w, g, state, lr, b1=0.9, b2=0.999, eps=1e-8, wd=0.0,
               decoupled=True):
    """E-9.10 and E-9.11.  One step of equations (9.8), (9.9) and (9.13).

    `state` is a dict you own; on the first call it will be empty, so create
    "m", "v" and "t" in it.  Mutate `w` in place and return it.

    decoupled=True is AdamW, equation (9.13): the decay is applied to w
    directly and never reaches the moments.  decoupled=False is the L2 form of
    D-9.3 step 1: the decay is added to the gradient before the moments see it.
    The two are not the same optimiser and E-9.11 measures the difference.

    The two branches differ in one line, and that line is the whole of D-9.3:
    `wd * w` either joins the gradient, in which case it is divided by
    sqrt(v_hat) along with everything else and the realised decay varies from
    coordinate to coordinate, or it is subtracted from w afterwards, in which
    case every coordinate decays by the same factor 1 - lr * wd.
    """
    state.setdefault("m", np.zeros_like(w))
    state.setdefault("v", np.zeros_like(w))
    state["t"] = state.get("t", 0) + 1
    t = state["t"]

    # L2 decay enters here, before the moments; decoupled decay does not.
    grad = g if decoupled else g + wd * w

    state["m"] = b1 * state["m"] + (1 - b1) * grad          # (9.8)
    state["v"] = b2 * state["v"] + (1 - b2) * grad ** 2     # (9.9)

    m_hat = state["m"] / (1 - b1 ** t)                      # bias correction
    v_hat = state["v"] / (1 - b2 ** t)

    w -= lr * m_hat / (np.sqrt(v_hat) + eps)
    if decoupled:
        w -= lr * wd * w                                    # (9.13)
    return w


def newton_schulz(G, steps=5, a=3.4445, b=-4.7750, c=2.0315):
    """E-9.12.  Equation (9.27), from X0 = G / ||G||_F.

    Return the iterate, which approximates U V^T for G = U S V^T.  Work on the
    smaller of G and G^T so the products are on the short side; the answer is
    the same either way, because the polynomial acts on the singular values and
    those do not care about the transpose.

    Each step is written as X <- a X + (b A + c A^2) X with A = X X^T, so the
    only square matrix ever formed is min(m, n) on a side.  For a 4096 x 1024
    gradient that is the difference between a 1024 x 1024 product and a
    4096 x 4096 one, for the same arithmetic.
    """
    X = np.asarray(G, dtype=np.float64)
    transposed = X.shape[0] > X.shape[1]
    if transposed:                      # keep A = X X^T on the short side
        X = X.T

    X = X / np.linalg.norm(X)           # X0 = G / ||G||_F, so ||X0||_2 <= 1

    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X

    return X.T if transposed else X


def peak_update_after_burst(b1, b2, eps=0.0, quiet=8000, g_quiet=1e-6,
                            g_burst=1.0):
    """E-9.13.  Run Adam's moment recursions with a quiet stretch of gradients
    of size g_quiet, then one gradient of size g_burst, and return the largest
    |m_hat / (sqrt(v_hat) + eps)| seen at or after the burst.

    Equation (9.21) says the answer is (1 - b1) / sqrt(1 - b2), independent of
    how big the burst was.

    The quiet stretch matters only in that it drives the bias corrections to
    one and leaves both moments negligible against the burst.  The peak of the
    burst response is then the burst step itself: on the next step m has been
    multiplied by b1 and sqrt(v) only by sqrt(b2), and b1 < sqrt(b2) for every
    pair anyone uses, so the ratio falls from there.  The tail is followed
    until it stops falling, which is where the burst has decayed out of both
    moments and the ratio is climbing back to the steady value of 1 that any
    constant gradient gives.  That later value measures the quiet stretch, not
    the burst, so the search stops at the turn.
    """
    m = v = 0.0
    for t in range(1, quiet + 1):                    # the quiet stretch
        m = b1 * m + (1 - b1) * g_quiet
        v = b2 * v + (1 - b2) * g_quiet * g_quiet

    def ratio(m, v, t):
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        return abs(m_hat / (np.sqrt(v_hat) + eps))

    t = quiet + 1                                    # the burst
    m = b1 * m + (1 - b1) * g_burst
    v = b2 * v + (1 - b2) * g_burst * g_burst
    peak = previous = ratio(m, v, t)

    while True:                                      # the decay back
        t += 1
        m = b1 * m + (1 - b1) * g_quiet
        v = b2 * v + (1 - b2) * g_quiet * g_quiet
        r = ratio(m, v, t)
        if r >= previous:                            # the turn: the burst is gone
            return float(peak)
        peak, previous = max(peak, r), r
