"""Your solutions for Chapter 14's [C] exercises.

THIS IS THE `solutions` BRANCH.  Every function below is worked in full and
every test in this directory passes.  On `main` these are stubs raising
NotImplementedError, and making them pass is the exercise; read this file only
after you have had a go, or when you want to compare a method rather than an
answer.  Appendix C prints the answer and the tolerance, not the code.
"""
import numpy as np
from scipy.special import logsumexp

from arith.decoding import tokens_per_round


# ------------------------------------------------------------------- E-14.3
def temper(z, T):
    """Softmax at temperature T, computed stably.

    `z` is a 1-D array of logits, `T > 0`.  Return `softmax(z / T)`.

    Subtract the maximum of z/T before exponentiating.  D-14.1's failure-mode
    note is about exactly this: fp16 logits spanning more than about 11
    saturate before the mathematics says they should.
    """
    u = np.asarray(z, dtype=np.float64) / float(T)
    e = np.exp(u - u.max())
    return e / e.sum()


def entropy(z, T):
    """H(p(T)) in nats, where p(T) = softmax(z / T).

    Note that 0 log 0 is 0.  A vector with a near-certain token will produce
    probabilities that underflow, and the naive expression returns nan.

    Method note.  Writing H = log Z(T) - E_p[z] / T, with log Z by log-sum-exp,
    removes the problem rather than patching it: no logarithm of a probability
    is ever taken, so an underflowed coordinate contributes an exact zero
    instead of a nan.  At an m-fold tie in the argmax it returns log m as
    T -> 0, which is the limit D-14.1 step 7 needs its genericity hypothesis
    for.
    """
    u = np.asarray(z, dtype=np.float64) / float(T)
    p = temper(z, T)
    return float(logsumexp(u) - p @ u)


def dentropy_dT(z, T):
    """The closed form (14.4): Var_{p(T)}(z) / T**3.

    The variance is taken under p(T), not under any fixed distribution, which
    is why dH/dT is not constant and the "randomness dial" intuition
    mis-calibrates.  Get the power right: T**2 agrees with T**3 at T = 1 and
    nowhere else.
    """
    z = np.asarray(z, dtype=np.float64)
    p = temper(z, T)
    mean = p @ z
    var = p @ (z - mean) ** 2
    return float(var / float(T) ** 3)


# ------------------------------------------------------------------ E-14.11
def top_k(p, k):
    """Keep the k largest coordinates of p and renormalise; zero the rest.

    Return an array of the same shape.  The zeros must be exact zeros, not
    small numbers: that is what makes truncation a projection onto a face
    rather than another reshaping.
    """
    p = np.asarray(p, dtype=np.float64)
    out = np.zeros_like(p)
    keep = np.argpartition(-p, k - 1)[:k]
    out[keep] = p[keep]
    return out / out.sum()


def top_p(p, thr):
    """Nucleus sampling, equation (14.8).

    Keep the smallest set S with sum_{i in S} p_i >= thr, renormalise, zero the
    rest.  A minimum-cardinality superlevel set is always an interval of the
    sorted order, so sort descending, take the cumulative sum, and stop at the
    first index that reaches the threshold: that index is INCLUDED.

    Off-by-one here is the classic bug and it is silent, because the result is
    still a valid distribution.
    """
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    # the first position whose cumulative mass reaches the threshold, included
    cut = int(np.searchsorted(cum, thr))
    cut = min(cut, len(p) - 1)
    out = np.zeros_like(p)
    keep = order[:cut + 1]
    out[keep] = p[keep]
    return out / out.sum()


def min_p(p, tau):
    """Keep {i : p_i >= tau * max(p)}, renormalise, zero the rest.

    Equation (14.9) shows this is a window in logit space of width
    T log(1/tau), so unlike top_p it never needs a sort or a cumulative sum.
    Implementing it directly on p is fine; knowing it is a logit window is what
    the exercise is for.
    """
    p = np.asarray(p, dtype=np.float64)
    out = np.where(p >= tau * p.max(), p, 0.0)
    return out / out.sum()


def compose(z, T, rule, arg, temperature_first=True):
    """Apply temperature and a truncation rule in the stated order.

    `rule` is one of "top_k", "top_p", "min_p".  With `temperature_first`,
    temper the logits and truncate the result, which is what (14.2) fixes as
    the book's order and what every serving stack does.  Otherwise truncate
    softmax(z) at T = 1 first, then re-temper the surviving coordinates and
    renormalise over the survivors.

    The two orders give different supports AND different odds among the
    survivors.  E-14.11 asks you to show that.
    """
    z = np.asarray(z, dtype=np.float64)
    truncate = {"top_k": top_k, "top_p": top_p, "min_p": min_p}[rule]
    if temperature_first:
        return truncate(temper(z, T), arg)
    keep = truncate(temper(z, 1.0), arg) > 0
    out = np.zeros_like(z)
    # re-tempering the survivors is a softmax over their logits alone
    out[keep] = temper(z[keep], T)
    return out


# ------------------------------------------------------------------ E-14.12
def beam_scores(candidates, rho=0.6):
    """Score a list of candidates under the three rules of D-14.2.

    Each candidate is a sequence of per-token log-probabilities.  Return a dict
    with keys "unnormalised", "mean", "lp", each mapping to a list of scores in
    the order the candidates were given, where

        unnormalised   S_n           = sum of the log-probabilities   (14.11)
        mean           S_n / n                                        (14.13)
        lp             S_n / lp(n),  lp(n) = ((5 + n) / 6) ** rho     (14.14)

    The 5 and the 6 are fitted constants from a machine-translation system.
    Nothing derives them.  That is the point of the exercise.
    """
    out = {"unnormalised": [], "mean": [], "lp": []}
    for c in candidates:
        n = len(c)
        s = float(np.sum(c))
        out["unnormalised"].append(s)
        out["mean"].append(s / n)
        out["lp"].append(s / ((5.0 + n) / 6.0) ** rho)
    return out


# ------------------------------------------------------------------ E-14.13
def _draw(dist, rng, n):
    """n independent draws from a categorical distribution, by inverse CDF."""
    return np.searchsorted(np.cumsum(dist), rng.random(n))


def speculative_emit(p, q, rng, n):
    """Draw n tokens by the sampler of D-14.3, and return them as an array.

    For each emission: draw x ~ q, accept with probability min(1, p(x)/q(x)),
    and on rejection draw instead from the residual p' proportional to
    (p - q)_+.

    The result is distributed exactly as p, for ANY q with the same support.
    E-14.13 asks you to demonstrate that by a chi-squared test, not to take it
    on trust.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    x = _draw(q, rng, n)
    accept = rng.random(n) < np.minimum(1.0, p[x] / q[x])
    residual = np.maximum(p - q, 0.0)
    total = residual.sum()
    # p == q accepts everything, and then the residual is never consulted
    residual = residual / total if total > 0 else p.copy()
    return np.where(accept, x, _draw(residual, rng, n))


def speculative_emit_broken(p, q, rng, n):
    """The same sampler with the residual replaced by p itself on rejection.

    This is the single most plausible wrong implementation, and it is wrong:
    derive its law and you get min(p, q) + TV(p, q) * p, which is not p.  Your
    chi-squared test should reject it decisively at the same sample size that
    leaves `speculative_emit` alone.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    x = _draw(q, rng, n)
    accept = rng.random(n) < np.minimum(1.0, p[x] / q[x])
    return np.where(accept, x, _draw(p, rng, n))


def expected_tokens(alpha, gamma):
    """(14.18).  The expected number of tokens emitted per verification round.

    Derive it rather than looking it up: P(N >= k) = alpha**k for k <= gamma, a
    tail sum gives E[N], and exactly one further token is emitted in either
    branch, which is why the sum runs from zero.

    The book prints this number, so it delegates to arith.decoding: the tail
    sum Sum_{k=0..gamma} alpha**k in closed form, with the alpha = 1 limit
    handled separately.
    """
    return tokens_per_round(float(alpha), int(gamma))
