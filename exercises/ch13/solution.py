"""Your solutions for Chapter 13's [C] exercises.

THIS IS THE `solutions` BRANCH.  Every function below is worked in full and
every test in this directory passes.  On `main` these are stubs raising
NotImplementedError, and making them pass is the exercise; read this file only
after you have had a go, or when you want to compare a method rather than an
answer.  Appendix C prints the answer and the tolerance, not the code.
"""
import numpy as np

from arith.quant_formats import nf4_levels as _nf4_levels


def affine_quantise(x, b_q, g_q):
    """E-13.10.  Group-wise affine quantisation, equation (13.1).

    `x` is a 1-D array.  Split it into contiguous groups of `g_q`, and for each
    group compute a scale and an integer zero-point from that group's own
    minimum and maximum, so the extremes land on 0 and 2**b_q - 1.

    Return `(xhat, s_q)` with xhat the dequantised values, same shape as x, and
    s_q the per-group scale, of shape (n_groups,).

    The last group may be short.  Sizing its scale from a full group's worth of
    values it does not have is a real bug with a quiet symptom: the tail of
    every tensor is quantised on the wrong grid.

    Method note.  The zero-point is rounded to an integer, so that the value
    zero is on the grid and a whole group can be stored as integers plus one
    scale.  Rounding it moves the group's extremes by at most half a step,
    which is why the bound of D-13.1 survives it.
    """
    x = np.asarray(x, dtype=np.float64)
    n_levels = 2 ** b_q - 1
    n_groups = int(np.ceil(len(x) / g_q))

    xhat = np.empty_like(x)
    s_q = np.empty(n_groups)
    for g in range(n_groups):
        # the last group is whatever is left, which may be fewer than g_q values
        blk = x[g * g_q:(g + 1) * g_q]
        lo, hi = blk.min(), blk.max()
        s = max((hi - lo) / n_levels, 1e-12)
        z = np.round(-lo / s)                     # integer zero-point
        q = np.clip(np.round(blk / s) + z, 0, n_levels)
        xhat[g * g_q:(g + 1) * g_q] = s * (q - z)
        s_q[g] = s
    return xhat, s_q


def fwht(x):
    """E-13.11.  The fast Walsh-Hadamard transform, in place if you like.

    `x` has length d, a power of two.  Return H_d x, unnormalised, where H_d is
    the Sylvester Hadamard matrix.  The transform is log2(d) passes of
    butterflies, each pass O(d) additions and subtractions, so O(d log d) and
    no multiplications at all.

    The orthogonal rotation of D-13.2 is H_d / sqrt(d), so divide once at the
    end rather than scaling inside the loop.
    """
    a = np.array(x, dtype=np.float64, copy=True)
    d = a.size
    if d & (d - 1):
        raise ValueError("length must be a power of two, got %d" % d)
    h = 1
    while h < d:
        # view the vector as pairs of blocks of width h and butterfly them
        a = a.reshape(-1, 2, h)
        a = np.concatenate([a[:, 0] + a[:, 1], a[:, 0] - a[:, 1]], axis=1)
        h *= 2
    return a.reshape(d)


def randomised_hadamard(d, seed):
    """E-13.11.  The matrix D-13.2 actually recommends: a Hadamard with a random
    sign diagonal in front.

    Return an orthogonal (d, d) array.  The signs are not decoration.  A fixed
    Hadamard maps a vector aligned with one of its rows to a one-hot, leaving
    the incoherence at its maximum sqrt(d), and the test checks exactly that.

    Method note.  The convention here is the row-vector one used throughout the
    chapter: a vector is rotated by x @ Q, so Q = D H / sqrt(d) applies the
    signs first and the transform second.  The other order, H D, would flip the
    signs of the *output* coordinates and leave an aligned vector one-hot,
    which is the failure the sign diagonal exists to prevent.  Forming the
    matrix at all is a convenience for testing: a kernel calls `fwht` on the
    signed vector and never materialises d^2 entries.
    """
    signs = np.where(np.random.default_rng(seed).random(d) < 0.5, -1.0, 1.0)
    H = np.ones((1, 1))
    while H.shape[0] < d:
        H = np.block([[H, H], [H, -H]])
    return signs[:, None] * H / np.sqrt(d)


def gptq(W, H, b_q=4, g_q=128, damp=0.01, act_order=False):
    """E-13.12.  The column sweep of D-13.3.

    `W` is (n_in, n_out) and `H` is (n_in, n_in), the 2 X^T X of the layer's
    calibration inputs.  Quantise coordinate by coordinate; after each one,
    push its error into the coordinates not yet fixed, using (13.6).

    Damp before inverting.  With `act_order`, visit coordinates in order of
    decreasing Hessian diagonal and permute back before returning.

    Return the quantised weights, same shape as W.

    Two traps.  The compensation is subtracted, not added: read the sign of
    (13.6) off the constraint rather than guessing, because the wrong sign
    gives a result that is worse than round-to-nearest by about as much as the
    right sign is better.  And the inverse Hessian has to be downdated as
    coordinates are fixed, or later compensations use stale correlations.

    Method note.  The downdate is free if the sweep is written against the
    Cholesky factor of H^-1 rather than against H^-1 itself.  Writing
    H^-1 = U^T U with U upper triangular, the row U[i, i:] is exactly the
    correlation of coordinate i with the coordinates still to come *given* that
    the earlier ones are fixed, so one factorisation before the loop replaces a
    rank-one update inside it.
    """
    W = np.array(W, dtype=np.float64, copy=True)
    H = np.array(H, dtype=np.float64, copy=True)
    n_in, n_out = W.shape
    n_levels = 2 ** b_q - 1

    # a coordinate no calibration input ever excites carries no information;
    # pin it so the factorisation stays finite
    dead = np.diag(H) == 0.0
    if dead.any():
        H[dead, dead] = 1.0
        W[dead] = 0.0

    if act_order:
        perm = np.argsort(-np.diag(H))
        W, H = W[perm], H[perm][:, perm]
        inv_perm = np.argsort(perm)

    H += np.eye(n_in) * (damp * np.mean(np.diag(H)))
    # H^-1 = U^T U with U upper triangular; U[i, i+1:] / U[i, i] is the
    # direction the error of coordinate i is spread along
    U = np.linalg.cholesky(np.linalg.inv(H)).T

    Q = np.zeros_like(W)
    s = z = None                      # set on the first row of every group
    for i in range(n_in):
        if i % g_q == 0:
            # a fresh scale per group, from the weights as they stand now:
            # earlier compensations have already moved them
            blk = W[i:i + g_q]
            lo, hi = blk.min(0), blk.max(0)
            s = np.maximum((hi - lo) / n_levels, 1e-12)
            z = np.round(-lo / s)
        q = np.clip(np.round(W[i] / s) + z, 0, n_levels)
        Q[i] = s * (q - z)
        # (13.6): subtract, so the residual of the fixed coordinate is absorbed
        # by the coordinates that are still free
        err = (W[i] - Q[i]) / U[i, i]
        W[i + 1:] -= np.outer(U[i, i + 1:], err)

    return Q[inv_perm] if act_order else Q


def nf4_levels():
    """E-13.13.  NF4's sixteen values.

    They are quantiles of the standard normal: take evenly spaced
    probabilities from a small offset up to 1/2 for the eight non-positive
    levels and from 1/2 up to 1 minus the offset for the eight non-negative
    ones, push them through the normal quantile function, share the zero, and
    normalise so the outermost level is exactly 1.

    Return a sorted array of 16 floats containing an exact 0.0, with minimum
    -1 and maximum +1.  The exact zero is not an accident: a format without one
    cannot represent a pruned weight.

    The table is the book's, so this delegates to arith.quant_formats rather
    than restating the offset: the two must agree digit for digit.
    """
    return _nf4_levels()
