"""Your solutions for Chapter 7's [C] exercises.

Every function raises NotImplementedError, so every test in this directory
fails on a fresh clone.  Making them pass is the exercise.  Work in float64
throughout: E-7.10's tolerance is four orders below what float32 can deliver.
"""


def softmax_backward(p, g):
    """E-7.8.  Equation (7.6), for a batch of rows.

    p is (..., n) with each row a probability vector, g is dL/dp of the same
    shape.  Return dL/dz.  Do not form the Jacobian; the whole point of the
    exercise is the O(n) working set per row.
    """
    raise NotImplementedError


def attention_backward(dO, Q, K, V, P, group, reduce="sum"):
    """E-7.9.  The masked grouped-query attention backward pass, D-7.3.

    Q  is (h, s, d_h);  K and V are (n_kv, s, d_h);  P is (h, s, s), already
    softmaxed and causally masked;  dO is (h, s, d_h).  `group` is h // n_kv.
    Return (dQ, dK, dV) with shapes (h, s, d_h), (n_kv, s, d_h), (n_kv, s, d_h).

    `reduce` decides how the group contributions to dK and dV are combined.
    "sum" is the correct answer, equation (7.22).  "mean" is the bug D-7.3's
    failure mode describes, and test_gradcheck asserts that it is caught.
    """
    raise NotImplementedError


def block_forward(x, W, cfg):
    """E-7.10.  One pre-norm block, equation (7.1).  Return (z, cache).

    W holds Q, K, V, O, gate, up, down, g1, g2.  cfg holds d, h, n_kv, d_h,
    d_ff.  Use RMSNorm with eps = 0 and a causal mask of -inf: E-7.10's
    tolerance does not survive an epsilon or a large-negative mask.
    """
    raise NotImplementedError


def block_backward(zbar, cache, W, cfg, reduce="sum"):
    """E-7.10.  The backward pass of block_forward.

    Return (xbar, grads) where grads is a dict with one entry per key of W,
    each the same shape as the weight it differentiates.
    """
    raise NotImplementedError
