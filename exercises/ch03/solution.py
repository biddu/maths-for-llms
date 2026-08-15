"""Your solutions for this chapter's [C] exercises.

Every function below raises NotImplementedError, so every test in this directory
fails on a fresh clone.  Making them pass is the exercise.  Worked solutions are
on the `solutions` branch; Appendix C prints the answer and the path, not the
code.
"""


def softmax_jacobian(z):
    """E-3.11.  Return diag(p) - p p^T for p = softmax(z)."""
    raise NotImplementedError


def causal_mha(X, Wq, Wk, Wv, Wo, h):
    """E-3.12.  Causal multi-head attention from (3.6) and (3.10)."""
    raise NotImplementedError


def qk_norm_logits(q, k):
    """E-3.13.  RMS-normalise q and k, then form the scaled inner product."""
    raise NotImplementedError


def linear_attention(q, k, v, phi=None):
    """E-3.14.  Equation (3.11) with a factorising feature map."""
    raise NotImplementedError
