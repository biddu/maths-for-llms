"""Your solutions for this chapter's [C] exercises.

THIS IS THE `solutions` BRANCH.  Every function below is worked in full and
every test in this directory passes.  On `main` these are stubs raising
NotImplementedError, and making them pass is the exercise; read this file only
after you have had a go, or when you want to compare a method rather than an
answer.  Appendix C prints the answer and the tolerance, not the code.
"""
import numpy as np


def _softmax(z):
    """Row-wise softmax, maximum subtracted, as in D-1.1 step 8."""
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def softmax_jacobian(z):
    """E-3.11.  Return diag(p) - p p^T for p = softmax(z)."""
    p = _softmax(np.asarray(z, dtype=float))
    return np.diag(p) - np.outer(p, p)


def causal_mha(X, Wq, Wk, Wv, Wo, h):
    """E-3.12.  Causal multi-head attention from (3.6) and (3.10).

    X is (s, d) with tokens as rows.  The three projections are (d, d); the
    per-head slices are contiguous column blocks of width d_h = d / h, which is
    what makes the concatenate-then-project of D-3.3 a block operation.  The
    mask is applied to the logits, before the softmax, so each row normalises
    over the positions it may see and no others.
    """
    X = np.asarray(X, dtype=float)
    s, d = X.shape
    d_h = d // h

    def heads(M):
        # (s, d) -> (h, s, d_h): head i owns columns [i*d_h, (i+1)*d_h).
        return (X @ M).reshape(s, h, d_h).transpose(1, 0, 2)

    q, k, v = heads(Wq), heads(Wk), heads(Wv)
    z = q @ k.transpose(0, 2, 1) / np.sqrt(d_h)          # (h, s, s)
    mask = np.tril(np.ones((s, s), dtype=bool))
    z = np.where(mask, z, -np.inf)
    a = _softmax(z)
    ctx = (a @ v).transpose(1, 0, 2).reshape(s, d)       # concatenate the heads
    return ctx @ np.asarray(Wo, dtype=float)


def qk_norm_logits(q, k):
    """E-3.13.  RMS-normalise q and k, then form the scaled inner product.

    RMS normalisation fixes ||q|| = ||k|| = sqrt(d_h), so the scaled inner
    product is exactly sqrt(d_h) cos theta: bounded by sqrt(d_h) whatever the
    entry variance has drifted to, which is the point of D-3.2's failure mode.

    The inner product is taken over the last axis under NumPy broadcasting, so
    q and k of the same shape give one logit per row pair, and q[:, None, :]
    against k[None, :, :] gives the full score matrix.
    """
    q = np.asarray(q, dtype=float)
    k = np.asarray(k, dtype=float)
    d_h = q.shape[-1]
    rms = lambda a: np.sqrt(np.mean(a * a, axis=-1, keepdims=True))
    qn, kn = q / rms(q), k / rms(k)
    return (qn * kn).sum(axis=-1) / np.sqrt(d_h)


def _elu_plus_one(x):
    """phi(x) = elu(x) + 1, strictly positive, so no denominator can vanish.

    The clamp inside the exponential only stops the branch `where` discards
    from overflowing on large positive entries.
    """
    return np.where(x > 0, x + 1.0, np.exp(np.minimum(x, 0.0)))


def linear_attention(q, k, v, phi=None):
    """E-3.14.  Equation (3.11) with a factorising feature map.

    Replace the kernel exp(<q, k>/sqrt(d_h)) by phi(q) phi(k)^T and the sums
    over keys leave the nonlinearity, so they can be accumulated once and
    reused by every query:

        out_i = phi(q_i) S / (phi(q_i) . z),  S = sum_j phi(k_j) v_j^T,
                                              z = sum_j phi(k_j).

    The state S is (m, d_v) and z is (m,), both independent of s, so the work
    is O(s m d_v) and no s x s matrix is ever formed.  The default feature map
    is phi(x) = elu(x) + 1, which is positive everywhere, so the denominator
    cannot vanish.
    """
    q, k, v = (np.asarray(a, dtype=float) for a in (q, k, v))
    pq, pk = (phi or _elu_plus_one)(q), (phi or _elu_plus_one)(k)
    S = pk.T @ v                        # (m, d_v), the whole of the keys
    z = pk.sum(axis=0)                  # (m,), the normaliser's state
    return (pq @ S) / (pq @ z)[:, None]
