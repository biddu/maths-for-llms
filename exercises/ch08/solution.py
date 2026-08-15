"""Your solutions for Chapter 8's [C] exercises.

Every function raises NotImplementedError, so every test in this directory
fails on a fresh clone.  Making them pass is the exercise.
"""


def cross_entropy_from_logits(logits, targets):
    """E-8.9.  Mean cross-entropy in nats, from raw logits.

    logits is (n, V), targets is (n,) of integer class indices.  Use the
    log-sum-exp shift: subtract each row's maximum before exponentiating.  The
    shift is exact, not an approximation, because softmax is shift-invariant
    (D-8.2 step 3), and it is the difference between a working loss and an
    overflow at a logit scale real models reach.
    """
    raise NotImplementedError


def bits_per_byte(ce_nats, bytes_per_token):
    """E-8.10.  Equation (8.7).  The one coordinate of Figure 8.1 that survives
    a change of tokenizer."""
    raise NotImplementedError


def perplexity(ce_nats):
    """E-8.10.  Equation (8.5)."""
    raise NotImplementedError


def causal_block_forward(X, W):
    """E-8.11.  One causal self-attention block, forward only.

    X is (s, d).  W is a dict with Q, K, V, O of shapes (d, d).  Single head,
    scaled dot product, additive -inf mask above the diagonal, then the output
    projection.  Return the (s, d) output.  No residual and no normalisation:
    the exercise is about the mask, not the block.
    """
    raise NotImplementedError


def token_losses(X, W, U, targets):
    """E-8.11.  Per-position cross-entropy from one parallel forward pass.

    Run causal_block_forward once on the whole sequence, project by U of shape
    (d, V), and return the (s,) vector of per-token losses in nats.  This is
    equation (8.17): all s of them, from one pass.
    """
    raise NotImplementedError
