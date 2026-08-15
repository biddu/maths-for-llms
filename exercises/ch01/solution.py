"""Your solutions for this chapter's [C] exercises.

Every function below raises NotImplementedError, so every test in this directory
fails on a fresh clone.  Making them pass is the exercise.  Worked solutions are
on the `solutions` branch; Appendix C prints the answer and the path, not the
code.
"""


def stable_softmax(z):
    """E-1.11.  Softmax via step 8 of D-1.1: subtract the row maximum first."""
    raise NotImplementedError


def count_params(L, d, h, d_h, n_kv, d_ff, V, tied):
    """E-1.12.  Total parameter count of a dense decoder-only model."""
    raise NotImplementedError


def linear_backward(X, W, dY):
    """E-1.13.  Return (dW, dX) for Y = XW under denominator layout."""
    raise NotImplementedError


def ce_grad(z, y):
    """E-1.14.  Gradient of softmax cross-entropy with respect to the logits."""
    raise NotImplementedError
