"""Your solutions for this chapter's [C] exercises.

THIS IS THE `solutions` BRANCH.  Every function below is worked in full and
every test in this directory passes.  On `main` these are stubs raising
NotImplementedError, and making them pass is the exercise; read this file only
after you have had a go, or when you want to compare a method rather than an
answer.  Appendix C prints the answer and the tolerance, not the code.
"""
import numpy as np

from arith.model_d import Config, total_params


def stable_softmax(z):
    """E-1.11.  Softmax via step 8 of D-1.1: subtract the row maximum first.

    Softmax is shift invariant, so subtracting the row maximum changes nothing
    mathematically and everything numerically: the largest exponent becomes
    exp(0) = 1, so no term can overflow, and the denominator is at least 1.
    """
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def count_params(L, d, h, d_h, n_kv, d_ff, V, tied):
    """E-1.12.  Total parameter count of a dense decoder-only model.

    The ledger itself lives in `arith/model_d.py`, which is what the book's
    arithmetic boxes print, so this counts by building that module's Config
    rather than by restating the formula.  For the record the formula is

        attention   d(h + 2 n_kv + h) d_h        GQA: K and V carry n_kv heads
        MLP         3 d d_ff                     SwiGLU: gate, up, down
        norms       2 d per layer, plus a final d
        embeddings  V d, twice unless tied
    """
    c = Config(L=L, d=d, h=h, d_h=d_h, n_kv=n_kv, d_ff=d_ff, V=V, tied=tied)
    return total_params(c)


def linear_backward(X, W, dY):
    """E-1.13.  Return (dW, dX) for Y = XW under denominator layout.

    D-1.4 step 7: each gradient has the shape of the tensor it is a gradient
    of, and that fixes both transposes without memorising either.
    """
    X, W, dY = np.asarray(X), np.asarray(W), np.asarray(dY)
    dW = X.T @ dY                       # (d_in, s) (s, d_out) = W's shape
    dX = dY @ W.T                       # (s, d_out) (d_out, d_in) = X's shape
    return dW, dX


def ce_grad(z, y):
    """E-1.14.  Gradient of softmax cross-entropy with respect to the logits.

    D-1.2: the softmax and the cross-entropy cancel down to p - y.  Note that
    the softmax is computed here, not passed in; a fused implementation is
    exactly what stops log p being evaluated at p near zero.
    """
    p = stable_softmax(z)
    return p - np.asarray(y, dtype=float)
