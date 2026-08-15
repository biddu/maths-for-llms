"""Your solutions for Chapter 11's [C] exercises.

THIS IS THE `solutions` BRANCH.  Every function below is worked in full and
every test in this directory passes.  On `main` these are stubs raising
NotImplementedError, and making them pass is the exercise; read this file only
after you have had a go, or when you want to compare a method rather than an
answer.  Appendix C prints the answer and the tolerance, not the code.

Three of the four are exact reformulations: the answer they compute is the same
answer the obvious implementation computes, and the tests say so with a
tolerance rather than an equality.  Read the docstring of
`test_online_softmax.py` for why that distinction is the point of the chapter
and not a concession.
"""
import numpy as np


def online_softmax_attention(z, v, n_blocks):
    """E-11.11.  Equations (11.12) to (11.14), the tiled softmax recurrence.

    `z` is a vector of s scores and `v` an (s, d_h) array of values.  Split the
    s positions into `n_blocks` contiguous blocks and run the recurrence,
    carrying only the running triple (m, l, o).  Return o / l, of shape (d_h,).

    Never form the full exp(z - max z) vector.  The whole point is that the
    only arrays that exist at once are one block's worth plus the accumulator,
    and an implementation that quietly materialises everything will pass the
    numerical test while missing the exercise.

    Two traps.  Initialise m so that exp(m - m_new) evaluates to 0 on the first
    block rather than to NaN: -inf works if you special-case it, a large finite
    negative works without a special case.  And rescale the *vector*
    accumulator by the same scalar factor as the denominator, which is correct
    by linearity and is step 6 of D-11.4.
    """
    z = np.asarray(z)
    v = np.asarray(v)
    s = z.shape[0]

    m = -np.finfo(np.float64).max        # finite, so exp(m - m_new) is 0 and not NaN
    l = 0.0                              # running denominator
    o = np.zeros(v.shape[1], dtype=np.float64)   # running numerator

    edges = np.linspace(0, s, n_blocks + 1).astype(int)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:                     # more blocks than positions
            continue
        zb, vb = z[lo:hi], v[lo:hi]      # the only block-sized arrays

        m_new = max(m, float(zb.max()))          # (11.12)
        rescale = np.exp(m - m_new)              # what the old triple is worth now
        p = np.exp(zb - m_new)                   # this block's unnormalised weights

        l = rescale * l + float(p.sum())         # (11.13)
        o = rescale * o + p @ vb                 # (11.14), the same scalar factor
        m = m_new

    return o / l


def mla_logits(X, C, W_q, W_uk, absorbed):
    """E-11.12.  The same logits by two routes, per D-11.2.

    X is (n_q, d) of residual rows, C is (n_k, d_c) of cached latents,
    W_q is (d, d_h) and W_uk is (d_c, d_h).

    absorbed=False: reconstruct K = C @ W_uk and return (X @ W_q) @ K.T, which
    is what a training-time implementation does.

    absorbed=True: fold W_q @ W_uk.T into one (d, d_c) matrix first, then
    return X @ that @ C.T.  K must never be formed.  Step 6 of D-11.2 is the
    entire content: matrix multiplication is associative.

    Return the (n_q, n_k) logits.  The test asserts the two agree and that the
    absorbed path never allocates an array of the materialised key's shape.
    """
    if not absorbed:
        K = C @ W_uk                     # (n_k, d_h), the materialised keys
        return (X @ W_q) @ K.T

    W_tilde = W_q @ W_uk.T               # (d, d_c), a load-time constant
    return (X @ W_tilde) @ C.T           # widest intermediate is (n_q, d_c)


def decoupled_rope_logits(x_i, x_j, c_j, W_q, W_uk, W_qr, W_kr, i, j,
                          base=10000.0):
    """E-11.13.  Equation (11.10), assembled from its two terms.

    x_i and x_j are the residual rows of the two tokens, c_j the cached latent
    of the second, and i and j their absolute positions.  Content and position
    are separate arguments on purpose: the property being tested is that moving
    the same two tokens to different absolute positions does not change the
    logit, and that is only a meaningful statement if the tokens can be held
    fixed while the positions move.

    Return the scalar

        (x_i @ W_q @ W_uk.T) @ c_j.T                 the compressed term
      + ((x_i @ W_qr) R_i) . ((x_j @ W_kr) R_j)      the rotated term

    scaled by 1/sqrt(d_h + d_r), where R_k is the RoPE matrix of §4.3 at
    absolute position k and d_r is the width of W_qr.

    Apply the rotation only to the second term.  Applying it to the first as
    well is the failure mode named in D-11.3, and it is silent: the numbers
    stay plausible and absorption is gone.

    The test checks that the result depends on i and j only through i - j,
    which is what the construction exists to guarantee.
    """
    d_h = W_q.shape[1]
    d_r = W_qr.shape[1]

    # the compressed term: no rotation, so W_q W_uk^T stays absorbable
    content = (x_i @ (W_q @ W_uk.T)) @ c_j

    # the decoupled term: two narrow vectors, each rotated at its own position
    q_r = _rope(x_i @ W_qr, i, base)
    k_r = _rope(x_j @ W_kr, j, base)
    rotated = float(q_r @ k_r)           # R_i R_j^T depends only on i - j

    return float((content + rotated) / np.sqrt(d_h + d_r))


def _rope(x, pos, base=10000.0):
    """§4.3's rotation applied to one vector of even width, in place of forming
    the (d_r, d_r) matrix.  Coordinates are paired (0,1), (2,3), ... and pair p
    turns by pos * base^(-2p/d_r)."""
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[-1]
    theta = pos * base ** (-np.arange(0, d, 2) / d)
    cos, sin = np.cos(theta), np.sin(theta)
    even, odd = x[0::2], x[1::2]
    out = np.empty_like(x)
    out[0::2] = even * cos - odd * sin
    out[1::2] = even * sin + odd * cos
    return out


def topk_recall(true_scores, approx_scores, k):
    """E-11.14.  How much of the true top-k a cheap surrogate finds.

    Both arrays are (n_queries, n_keys) and may contain -inf where a causal
    mask applies.  For each row with at least k unmasked entries, take the true
    top-k set and the approximate top-k set, and return the mean over rows of
    |intersection| / k.

    Rows with fewer than k valid keys have no meaningful top-k and must be
    skipped, not scored as perfect.  Getting that wrong inflates recall at
    small k, which is exactly where the number is being read.

    Recall is a property of the *selection*, not of the output.  A missed key
    with a small attention weight costs nothing; a missed key with a large one
    may cost everything, and this statistic cannot tell you which happened.
    The exercise asks you to say so.

    Any leading axes are flattened into the query axis, so a stack of per-head
    or per-layer score matrices is scored as one pool of rows.  That is the
    right pooling for this statistic: every row is one query's selection
    problem, whichever head posed it.
    """
    true = np.asarray(true_scores, dtype=np.float64).reshape(-1, np.shape(true_scores)[-1])
    approx = np.asarray(approx_scores, dtype=np.float64).reshape(true.shape)

    hits = []
    for t_row, a_row in zip(true, approx):
        valid = np.isfinite(t_row)
        if valid.sum() < k:              # no meaningful top-k: skip, do not score
            continue
        top_true = np.argpartition(-t_row, k - 1)[:k]
        top_approx = np.argpartition(-a_row, k - 1)[:k]
        hits.append(len(np.intersect1d(top_true, top_approx, assume_unique=True)) / k)

    return float(np.mean(hits)) if hits else 0.0
