"""Your solutions for Chapter 7's [C] exercises.

THIS IS THE `solutions` BRANCH.  Every function below is worked in full and
every test in this directory passes.  On `main` these are stubs raising
NotImplementedError, and making them pass is the exercise; read this file only
after you have had a go, or when you want to compare a method rather than an
answer.  Appendix C prints the answer and the tolerance, not the code.

Work in float64 throughout: E-7.10's tolerance is four orders below what float32 can deliver.
"""

import numpy as np
from scipy.special import expit


def softmax_backward(p, g):
    """E-7.8.  Equation (7.6), for a batch of rows.

    p is (..., n) with each row a probability vector, g is dL/dp of the same
    shape.  Return dL/dz.  Do not form the Jacobian; the whole point of the
    exercise is the O(n) working set per row.

    (diag(p) - p p^T) g = p * (g - p.g), so the row needs one dot product and
    one subtraction.  Every entry of the row is shifted by the same p.g, which
    is D-7.2 step 7: only the contrast in g survives.
    """
    p = np.asarray(p, dtype=float)
    g = np.asarray(g, dtype=float)
    return p * (g - (p * g).sum(axis=-1, keepdims=True))


def attention_backward(dO, Q, K, V, P, group, reduce="sum"):
    """E-7.9.  The masked grouped-query attention backward pass, D-7.3.

    Q  is (h, s, d_h);  K and V are (n_kv, s, d_h);  P is (h, s, s), already
    softmaxed and causally masked;  dO is (h, s, d_h).  `group` is h // n_kv.
    Return (dQ, dK, dV) with shapes (h, s, d_h), (n_kv, s, d_h), (n_kv, s, d_h).

    `reduce` decides how the group contributions to dK and dV are combined.
    "sum" is the correct answer, equation (7.22).  "mean" is the bug D-7.3's
    failure mode describes, and test_gradcheck asserts that it is caught.

    The masked entries need no attention of their own: an -inf mask leaves
    P = 0 there exactly, and softmax_backward multiplies by P, so dS is zero
    above the diagonal without a second masking step.
    """
    dO = np.asarray(dO, dtype=float)
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
    P = np.asarray(P, dtype=float)
    if reduce not in ("sum", "mean"):
        raise ValueError("reduce must be 'sum' or 'mean', not %r" % (reduce,))

    h, s, d_h = Q.shape
    scale = 1.0 / np.sqrt(d_h)
    dQ = np.zeros_like(Q)
    dK = np.zeros_like(K)
    dV = np.zeros_like(V)

    for i in range(h):
        kv = i // group
        dV[kv] += P[i].T @ dO[i]              # O = P V, broadcast over the group
        dP = dO[i] @ V[kv].T
        dS = softmax_backward(P[i], dP)       # equation (7.6), row by row
        dQ[i] = (dS @ K[kv]) * scale          # S = Q K^T / sqrt(d_h)
        dK[kv] += (dS.T @ Q[i]) * scale

    if reduce == "mean":                      # the planted bug, not the answer
        dK /= group
        dV /= group
    return dQ, dK, dV


def _rmsnorm_forward(x, g):
    """RMSNorm with eps = 0, returning what the backward pass needs.

    Return (y, xhat, r): the normalised-and-scaled output, the unit-RMS
    direction and the per-row RMS.
    """
    r = np.sqrt((x * x).mean(axis=-1, keepdims=True))
    xhat = x / r
    return xhat * g, xhat, r


def _rmsnorm_backward(dy, xhat, r, g):
    """The Jacobian of D-5.1 at eps = 0.  Return (dx, dg).

    dx = (g * dy - xhat (xhat . (g * dy)) / d) / r, so the radial component is
    removed exactly: xhat . xhat is d when eps is zero, and E-7.10's tolerance
    is what makes that exactness worth having.
    """
    d = xhat.shape[-1]
    gdy = g * dy
    dx = (gdy - xhat * (xhat * gdy).sum(axis=-1, keepdims=True) / d) / r
    dg = (dy * xhat).sum(axis=0)
    return dx, dg


def _split_heads(a, n_heads, d_h):
    """(s, n_heads * d_h) -> (n_heads, s, d_h)."""
    s = a.shape[0]
    return a.reshape(s, n_heads, d_h).transpose(1, 0, 2)


def _merge_heads(a):
    """(n_heads, s, d_h) -> (s, n_heads * d_h), the inverse of _split_heads."""
    n_heads, s, d_h = a.shape
    return a.transpose(1, 0, 2).reshape(s, n_heads * d_h)


def _attention_forward(Qh, Kh, Vh, group):
    """Masked grouped-query attention, equation (7.15).  Return (A, P).

    The mask is -inf and the row maximum is subtracted before exponentiating,
    which is the same shift as D-8.2 step 3 and is exact.
    """
    h, s, d_h = Qh.shape
    mask = np.triu(np.full((s, s), -np.inf), 1)
    P = np.empty((h, s, s))
    A = np.empty((h, s, d_h))
    for i in range(h):
        kv = i // group
        S = Qh[i] @ Kh[kv].T / np.sqrt(d_h) + mask
        S = S - S.max(axis=-1, keepdims=True)
        e = np.exp(S)
        P[i] = e / e.sum(axis=-1, keepdims=True)
        A[i] = P[i] @ Vh[kv]
    return A, P


def block_forward(x, W, cfg):
    """E-7.10.  One pre-norm block, equation (7.1).  Return (z, cache).

    W holds Q, K, V, O, gate, up, down, g1, g2.  cfg holds d, h, n_kv, d_h,
    d_ff.  Use RMSNorm with eps = 0 and a causal mask of -inf: E-7.10's
    tolerance does not survive an epsilon or a large-negative mask.

    The block is h = x + Attn(RMSNorm(x)) then z = h + FFN(RMSNorm(h)), with
    the FFN of equation (6.7).  Everything the backward pass would otherwise
    recompute is kept in the cache; that trade, memory against recomputation,
    is section 7.7's whole subject.
    """
    x = np.asarray(x, dtype=float)
    h, n_kv, d_h = cfg["h"], cfg["n_kv"], cfg["d_h"]
    group = h // n_kv

    x1, xhat1, r1 = _rmsnorm_forward(x, W["g1"])
    Qh = _split_heads(x1 @ W["Q"], h, d_h)
    Kh = _split_heads(x1 @ W["K"], n_kv, d_h)
    Vh = _split_heads(x1 @ W["V"], n_kv, d_h)
    A, P = _attention_forward(Qh, Kh, Vh, group)
    cat = _merge_heads(A)
    res = x + cat @ W["O"]

    x2, xhat2, r2 = _rmsnorm_forward(res, W["g2"])
    gate_pre = x2 @ W["gate"]
    up_pre = x2 @ W["up"]
    sig = expit(gate_pre)
    act = gate_pre * sig * up_pre               # SwiGLU, equation (6.7)
    z = res + act @ W["down"]

    cache = {"x": x, "xhat1": xhat1, "r1": r1, "x1": x1,
             "Qh": Qh, "Kh": Kh, "Vh": Vh, "P": P, "cat": cat, "res": res,
             "xhat2": xhat2, "r2": r2, "x2": x2,
             "gate_pre": gate_pre, "up_pre": up_pre, "sig": sig, "act": act}
    return z, cache


def block_backward(zbar, cache, W, cfg, reduce="sum"):
    """E-7.10.  The backward pass of block_forward.

    Return (xbar, grads) where grads is a dict with one entry per key of W,
    each the same shape as the weight it differentiates.

    Read it bottom-up against block_forward.  Each residual junction forks the
    incoming gradient: the branch that goes through the sublayer and the copy
    that skips it, added back at the end.
    """
    zbar = np.asarray(zbar, dtype=float)
    h, n_kv, d_h = cfg["h"], cfg["n_kv"], cfg["d_h"]
    group = h // n_kv
    grads = {}

    # --- the FFN, and the residual that skips it -------------------------
    grads["down"] = cache["act"].T @ zbar
    dact = zbar @ W["down"].T
    sig, gate_pre, up_pre = cache["sig"], cache["gate_pre"], cache["up_pre"]
    dsilu = sig * (1.0 + gate_pre * (1.0 - sig))     # d/du [u sigma(u)]
    dgate_pre = dact * up_pre * dsilu
    dup_pre = dact * gate_pre * sig
    grads["gate"] = cache["x2"].T @ dgate_pre
    grads["up"] = cache["x2"].T @ dup_pre
    dx2 = dgate_pre @ W["gate"].T + dup_pre @ W["up"].T
    dres, grads["g2"] = _rmsnorm_backward(dx2, cache["xhat2"], cache["r2"],
                                          W["g2"])
    dres = dres + zbar

    # --- attention, and the residual that skips it -----------------------
    grads["O"] = cache["cat"].T @ dres
    dA = _split_heads(dres @ W["O"].T, h, d_h)
    dQh, dKh, dVh = attention_backward(dA, cache["Qh"], cache["Kh"],
                                       cache["Vh"], cache["P"], group,
                                       reduce=reduce)
    dq, dk, dv = _merge_heads(dQh), _merge_heads(dKh), _merge_heads(dVh)
    grads["Q"] = cache["x1"].T @ dq
    grads["K"] = cache["x1"].T @ dk
    grads["V"] = cache["x1"].T @ dv
    dx1 = dq @ W["Q"].T + dk @ W["K"].T + dv @ W["V"].T
    dx, grads["g1"] = _rmsnorm_backward(dx1, cache["xhat1"], cache["r1"],
                                        W["g1"])
    xbar = dx + dres
    return xbar, grads
