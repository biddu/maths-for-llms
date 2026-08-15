"""Your solutions for Chapter 12's [C] exercises.

THIS IS THE `solutions` BRANCH.  Every function below is worked in full and
every test in this directory passes.  On `main` these are stubs raising
NotImplementedError, and making them pass is the exercise; read this file only
after you have had a go, or when you want to compare a method rather than an
answer.  Appendix C prints the answer and the tolerance, not the code.

The three are the chapter in miniature: a router that makes a discrete choice
differentiable, a loss that measures how badly that choice is distributed, and
a controller that fixes the distribution without touching the objective.
"""
import numpy as np


def _softmax(z):
    """Row-wise softmax, max-subtracted.  (12.1) as written is exp z / sum exp
    z; subtracting the row maximum first is the same function and is what
    §8.4 requires of any implementation of it."""
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def route(z, k):
    """E-12.9.  Top-k routing with renormalisation, equations (12.1) and (12.2).

    `z` is (T, E) of router logits.  Return `(idx, ghat)`, both (T, k):
    the indices of each token's k largest logits, and the softmax gates of
    those experts renormalised to sum to one over the selected set.

    Two things the tests check and one they cannot.  The selected gates must
    sum to one per token, and the support must have size exactly k.  What no
    test can check is that you renormalised over the *selected* set rather
    than over all E: dividing by the full softmax sum leaves gates summing to
    less than one, which trains, and quietly destroys D-12.1.

    Selection is on the logits, by partial sort, and never on the gates.  The
    two orderings agree here because the softmax is monotone, but the bias of
    (12.8) is added to the logits for selection only, and at that point an
    implementation that ranked gates would be ranking the wrong array.
    """
    z = np.asarray(z, dtype=np.float64)
    idx = np.argpartition(-z, k - 1, axis=1)[:, :k]      # the k largest logits
    # order the selection by decreasing logit, so the columns mean something
    order = np.argsort(-np.take_along_axis(z, idx, axis=1), axis=1)
    idx = np.take_along_axis(idx, order, axis=1)

    g = _softmax(z)                                      # (12.1), over all E
    sel = np.take_along_axis(g, idx, axis=1)
    ghat = sel / sel.sum(axis=1, keepdims=True)          # (12.2), over the k
    return idx, ghat


def aux_loss(z, idx, alpha_aux):
    """E-12.10.  The auxiliary loss of (12.6) and its gradient.

    `z` is (T, E) of logits, `idx` the (T, k) selection from `route`.  Return
    `(loss, grad)` with grad the same shape as z.

    Build f as the fraction of the T*k assignments that went to each expert,
    and P as the mean softmax gate.  The loss is alpha_aux * E * <f, P>.

    f is a *count*.  It is piecewise constant in z and carries no gradient at
    all: every derivative flows through P, and an implementation that lets f
    carry gradient will disagree with finite differences and will also be
    minimising something other than what (12.6) says.  The analytic gradient
    is (12.7); deriving it is E-12.1.
    """
    z = np.asarray(z, dtype=np.float64)
    idx = np.asarray(idx)
    T, E = z.shape

    f = np.bincount(idx.ravel(), minlength=E) / idx.size     # the load, a count
    g = _softmax(z)
    P = g.mean(axis=0)                                       # the mean gate

    loss = float(alpha_aux * E * (f @ P))

    # (12.7).  The softmax Jacobian applied to f: expert j is pushed down when
    # its load exceeds this token's own probability-weighted mean load.
    weighted_mean = (g * f[None, :]).sum(axis=1, keepdims=True)
    grad = (alpha_aux * E / T) * g * (f[None, :] - weighted_mean)
    return loss, grad


def bias_controller(loads_fn, E, k, u, steps, gamma0=None):
    """E-12.11.  The bias-adjusted routing controller of (12.8).

    `loads_fn(gamma)` returns the (E,) vector of per-expert load fractions
    that result from selecting on z + gamma; it stands in for a training step.
    Run `steps` iterations of

        gamma <- gamma + u * sign(1/E - load)

    starting from `gamma0` (zeros if None).  Return `(gamma, history)` where
    history is (steps, E) of the load errors 1/E - load, one row per step.

    Note what is *not* here: no gradient, no optimiser, no autograd.  The bias
    is outside the graph by construction, which is step 1 of D-12.3 and the
    whole reason the scheme costs the objective nothing.

    Each row of the history is the error measured at the gamma that step
    started from, so row 0 is the imbalance of the unbiased router.  The step
    is a fixed magnitude u in the sign direction, which is why the error does
    not converge to zero but to a limit cycle of amplitude of order u times the
    plant gain: step 7 of D-12.3.
    """
    gamma = np.zeros(E, dtype=np.float64) if gamma0 is None \
        else np.array(gamma0, dtype=np.float64)
    target = 1.0 / E

    history = np.empty((steps, E), dtype=np.float64)
    for t in range(steps):
        error = target - np.asarray(loads_fn(gamma), dtype=np.float64)
        history[t] = error
        gamma = gamma + u * np.sign(error)           # (12.8), sign not gradient

    return gamma, history
