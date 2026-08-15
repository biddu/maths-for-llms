"""Your solutions for Chapter 8's [C] exercises.

THIS IS THE `solutions` BRANCH.  Every function below is worked in full and
every test in this directory passes.  On `main` these are stubs raising
NotImplementedError, and making them pass is the exercise; read this file only
after you have had a go, or when you want to compare a method rather than an
answer.  Appendix C prints the answer and the tolerance, not the code.
"""

import numpy as np

from arith.model_d import loss_units


def _row_losses(logits, targets):
    """Per-row cross-entropy in nats, log-sum-exp shifted.

    Shared by the two exercises below, because the s losses of equation (8.17)
    and the mean of equation (8.2) differ only in whether you average.
    """
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets)
    z = logits - logits.max(axis=-1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=-1))
    true = np.take_along_axis(z, targets[..., None], axis=-1)[..., 0]
    return lse - true


def cross_entropy_from_logits(logits, targets):
    """E-8.9.  Mean cross-entropy in nats, from raw logits.

    logits is (n, V), targets is (n,) of integer class indices.  Use the
    log-sum-exp shift: subtract each row's maximum before exponentiating.  The
    shift is exact, not an approximation, because softmax is shift-invariant
    (D-8.2 step 3), and it is the difference between a working loss and an
    overflow at a logit scale real models reach.

    After the shift the largest exponent is exp(0) = 1 and the rest underflow
    towards zero, so the sum is between 1 and V and cannot overflow whatever
    the logits were.
    """
    return float(_row_losses(logits, targets).mean())


def bits_per_byte(ce_nats, bytes_per_token):
    """E-8.10.  Equation (8.7).  The one coordinate of Figure 8.1 that survives
    a change of tokenizer.

    Nats become bits by dividing by ln 2, and bits per token become bits per
    byte by dividing by the tokenizer's measured bytes per token.  The
    conversion is arith/model_d.py's loss_units, which is what prints the
    chapter's four coordinates, so the two cannot drift apart.
    """
    return loss_units(ce_nats, bytes_per_token)["bits_per_byte"]


def perplexity(ce_nats):
    """E-8.10.  Equation (8.5)."""
    return loss_units(ce_nats)["perplexity"]


def causal_block_forward(X, W):
    """E-8.11.  One causal self-attention block, forward only.

    X is (s, d).  W is a dict with Q, K, V, O of shapes (d, d).  Single head,
    scaled dot product, additive -inf mask above the diagonal, then the output
    projection.  Return the (s, d) output.  No residual and no normalisation:
    the exercise is about the mask, not the block.
    """
    X = np.asarray(X, dtype=float)
    s, d = X.shape
    q, k, v = X @ W["Q"], X @ W["K"], X @ W["V"]
    S = q @ k.T / np.sqrt(d) + np.triu(np.full((s, s), -np.inf), 1)
    S = S - S.max(axis=-1, keepdims=True)
    e = np.exp(S)                      # exp(-inf) is 0 exactly, so row i sees
    P = e / e.sum(axis=-1, keepdims=True)   # only positions 0 .. i
    return (P @ v) @ W["O"]


def token_losses(X, W, U, targets):
    """E-8.11.  Per-position cross-entropy from one parallel forward pass.

    Run causal_block_forward once on the whole sequence, project by U of shape
    (d, V), and return the (s,) vector of per-token losses in nats.  This is
    equation (8.17): all s of them, from one pass.

    Row i of the output already depends on positions 0 .. i and nothing later,
    so the s prefixes a sequential reader would run one at a time are all
    present in the single (s, s) attention matrix.  The mask removes a
    dependency, not a computation.
    """
    out = causal_block_forward(X, W)
    return _row_losses(out @ U, np.asarray(targets))
