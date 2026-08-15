"""Your solutions for Chapter 15's [C] exercises.

THIS IS THE `solutions` BRANCH.  Every function below is worked in full and
every test in this directory passes.  On `main` these are stubs raising
NotImplementedError, and making them pass is the exercise; read this file only
after you have had a go, or when you want to compare a method rather than an
answer.  Appendix C prints the answer and the tolerance, not the code.
"""
import numpy as np
from scipy.special import expit, logsumexp


# ------------------------------------------------------------------ E-15.11
def _scorer_forward(params, X):
    """s(X) = tanh(tanh(X @ W1) @ W2) @ w3, keeping both hidden layers."""
    W1, W2, w3 = params
    h1 = np.tanh(X @ W1)
    h2 = np.tanh(h1 @ W2)
    return h1, h2, h2 @ w3


def _scorer_backward(params, X, h1, h2, c):
    """Gradients of sum_i c_i s(X_i), with the forward pass supplied."""
    _, W2, w3 = params
    g3 = h2.T @ c
    da2 = np.outer(c, w3) * (1.0 - h2 ** 2)
    g2 = h1.T @ da2
    da1 = (da2 @ W2.T) * (1.0 - h1 ** 2)
    g1 = X.T @ da1
    return [g1, g2, g3]


def rm_loss_and_grad(params, Xw, Xl):
    """The Bradley-Terry reward-model NLL (15.4) and its gradient (15.5).

    `params` is a list [W1, W2, w3] of a three-layer tanh scorer:
        s(X) = tanh(tanh(X @ W1) @ W2) @ w3
    `Xw` and `Xl` are (n, d) arrays of features for the chosen and rejected
    completions.  Return `(loss, grads)` with `grads` a list matching `params`.

    The whole content of (15.5) is the per-pair weight sigma(-Delta), which is
    the SAME scorer's margin.  Get that factor right and the rest is the chain
    rule; get it wrong and the loss still falls, which is why this is checked
    against a numerical derivative rather than against intuition.

    Method note.  The loss is averaged over pairs, and -log sigma(Delta) is
    evaluated as softplus(-Delta) so that a confidently wrong pair costs its
    margin rather than overflowing.  The two branches share parameters, so the
    chosen features are backpropagated with weight sigma(-Delta) and the
    rejected ones with the same weight negated: one scorer, two passes.
    """
    Xw = np.asarray(Xw, dtype=np.float64)
    Xl = np.asarray(Xl, dtype=np.float64)
    n = Xw.shape[0]

    h1w, h2w, sw = _scorer_forward(params, Xw)
    h1l, h2l, sl = _scorer_forward(params, Xl)
    delta = sw - sl

    loss = float(np.mean(np.logaddexp(0.0, -delta)))
    # dL/dDelta = -sigma(-Delta)/n: the gain vanishes on pairs already ranked
    # correctly and confidently, which is D-15.2's self-annealing
    c = -expit(-delta) / n
    gw = _scorer_backward(params, Xw, h1w, h2w, c)
    gl = _scorer_backward(params, Xl, h1l, h2l, -c)
    return loss, [a + b for a, b in zip(gw, gl)]


# ------------------------------------------------------------------ E-15.12
def _tilted_logits(pi_ref, r, beta):
    """log pi_ref + r / beta, with log 0 = -inf kept as -inf.

    Coordinates outside the reference's support stay at -inf whatever the
    reward is, which is the absolute-continuity clause of D-15.3 in one line.
    """
    pi_ref = np.asarray(pi_ref, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return np.where(pi_ref > 0, np.log(pi_ref) + r / float(beta), -np.inf)


def pi_star(pi_ref, r, beta):
    """(15.7).  The KL-regularised optimum, computed stably.

    Return the distribution proportional to pi_ref * exp(r / beta).  Work in
    log space: r/beta overflows for small beta long before the answer does.
    """
    u = _tilted_logits(pi_ref, r, beta)
    e = np.exp(u - u.max())
    return e / e.sum()


def objective(pi, pi_ref, r, beta):
    """(15.5).  J(pi) = E_pi[r] - beta KL(pi || pi_ref).

    Adopt 0 log 0 = 0.  If pi puts mass where pi_ref does not, J is -infinity;
    that is D-15.3's absolute-continuity clause and not an edge case to paper
    over.
    """
    pi = np.asarray(pi, dtype=np.float64)
    pi_ref = np.asarray(pi_ref, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    live = pi > 0
    if np.any(live & (pi_ref <= 0)):
        return -np.inf
    kl = float(np.sum(pi[live] * np.log(pi[live] / pi_ref[live])))
    return float(pi @ r - float(beta) * kl)


def log_Z(pi_ref, r, beta):
    """log of the partition function in (15.7), by log-sum-exp.

    beta * log_Z is the optimal value of J, and it is a free energy: it tends
    to max(r) as beta -> 0 and to E_ref[r] as beta -> infinity.  Checking both
    limits is a good way to find a sign error.
    """
    return float(logsumexp(_tilted_logits(pi_ref, r, beta)))


# ------------------------------------------------------------------ E-15.13
def dpo_step(theta, phi, pi_ref, beta, lr):
    """One gradient step on (15.12) for a featured tabular model.

    Completion i has feature vector `phi[i]` and logit `theta @ phi[i]`, so the
    policy is softmax over those logits.  Completion 0 is chosen and 1 is
    rejected; any others are unlabelled and appear only through the softmax
    normaliser, which is where the displaced mass goes.

    Return `(theta_new, loss, pi)` with `pi` the policy BEFORE the step.

    Method note.  Write the implicit margin as
    h = beta[(log pi_w - log pi_ref_w) - (log pi_l - log pi_ref_l)].  Each
    grad log pi_i is phi_i minus the policy-averaged feature, so that average
    cancels in the difference and grad h = beta (phi_w - phi_l).  The step
    therefore moves theta along one fixed direction, and nothing in it asks
    where the displaced probability goes: an unlabelled completion far along
    phi_w - phi_l collects all of it, and both labelled completions lose mass
    while the loss falls.  That is likelihood displacement.
    """
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    pi_ref = np.asarray(pi_ref, dtype=np.float64)

    u = phi @ theta
    e = np.exp(u - u.max())
    pi = e / e.sum()

    logratio = np.log(pi) - np.log(pi_ref)
    h = float(beta) * (logratio[0] - logratio[1])
    loss = float(np.logaddexp(0.0, -h))

    phibar = pi @ phi                       # the policy-averaged feature
    grad_h = float(beta) * ((phi[0] - phibar) - (phi[1] - phibar))
    grad = -expit(-h) * grad_h              # dL/dtheta
    return theta - float(lr) * grad, loss, pi


# ------------------------------------------------------------------ E-15.14
def ratio_variances(n_trials, length, p_flip, jump, rng, all_tokens=True):
    """Simulate routing flips and return (var_token, var_sequence).

    Each token's log-ratio is `jump` with probability `p_flip` and 0 otherwise.
    The token statistic is one fixed position's log-ratio; the sequence
    statistic is (1/|y|) times the sum over all positions, which is the log of
    (15.18).

    With `all_tokens=False` only position 0 may flip.  The two settings give
    variance ratios of exactly |y| and |y|^2 respectively, and E-15.14 asks you
    to derive both before measuring them.

    Method note.  Both statistics are read off the SAME simulated sequences,
    which is what makes the second ratio exact rather than merely expected: if
    only position 0 can flip then the sequence statistic is the token statistic
    divided by |y|, sample by sample.
    """
    n_trials, length = int(n_trials), int(length)
    x = np.zeros((n_trials, length))
    n_flippable = length if all_tokens else 1
    flip = rng.random((n_trials, n_flippable)) < p_flip
    x[:, :n_flippable] = np.where(flip, float(jump), 0.0)

    token = x[:, 0]                          # one fixed position
    sequence = x.mean(axis=1)                # (1/|y|) sum over positions
    return float(np.var(token)), float(np.var(sequence))
