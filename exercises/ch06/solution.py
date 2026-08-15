"""Your solutions for Chapter 6's [C] exercises.

THIS IS THE `solutions` BRANCH.  Every function below is worked in full and
every test in this directory passes.  On `main` these are stubs raising
NotImplementedError, and making them pass is the exercise; read this file only
after you have had a go, or when you want to compare a method rather than an
answer.  Appendix C prints the answer and the tolerance, not the code.
"""

import numpy as np
from scipy.special import expit

from arith.model_d import llama_intermediate_size


def swiglu(x, W_gate, W_up, W_down, beta=1.0):
    """E-6.8.  Equation (6.7), row-vector convention.

    x is (n, d); W_gate and W_up are (d, d_ff); W_down is (d_ff, d).
    Return the (n, d) output.

    The gate is the sigmoid-weighted linear unit z sigma(beta z), computed with
    scipy's expit rather than 1/(1+exp(-beta z)) so that a large negative beta z
    underflows to zero instead of overflowing the exponential.
    """
    x = np.asarray(x, dtype=float)
    z = x @ np.asarray(W_gate, dtype=float)
    u = x @ np.asarray(W_up, dtype=float)
    return (z * expit(beta * z) * u) @ np.asarray(W_down, dtype=float)


def intermediate_size(d, multiplier=1.3, multiple_of=1024):
    """E-6.9.  The four-step width pipeline of the Chapter 6 arithmetic box.

    4d, then two thirds truncated, then times the multiplier truncated, then
    rounded *up* to a multiple of multiple_of.  Order matters and so does the
    direction of each rounding.

    The pipeline itself lives in arith/model_d.py, which is what regenerates
    the printed box, so it is imported rather than transcribed: two copies of
    four lines is one copy too many.
    """
    return int(llama_intermediate_size(d, multiplier, multiple_of)
               ["intermediate_size"])


def gated_basis(Q, n):
    """E-6.10.  The n rank-2 Hessians a gated layer of width n contributes.

    Q is (d, d) with orthonormal columns.  Use equation (6.13): pair column j
    with column d-1-j, set w = u_p + u_q and v = u_p - u_q, and return the list
    of sym(w v^T) for j = 0 .. n-1.

    sym(w v^T) collapses to u_p u_p^T - u_q u_q^T, so each gated unit carries
    one positive and one negative curvature direction: that is the rank two of
    D-6.3 step 6, and the reason width n reaches rank 2n.
    """
    Q = np.asarray(Q, dtype=float)
    d = Q.shape[1]
    basis = []
    for j in range(n):
        u_p, u_q = Q[:, j], Q[:, d - 1 - j]
        w, v = u_p + u_q, u_p - u_q
        B = np.outer(w, v)
        basis.append(0.5 * (B + B.T))
    return basis


def best_fit(basis, S):
    """E-6.10.  Relative residual of the least-squares fit of S in the span of
    the given symmetric matrices.  Return
    ||sum_j c_j B_j - S||_F / ||S||_F at the optimal c.

    Vectorising each B_j turns the Frobenius fit into an ordinary linear least
    squares, and lstsq's minimum-norm solution keeps the answer well defined
    when the basis is linearly dependent.
    """
    S = np.asarray(S, dtype=float)
    A = np.column_stack([np.asarray(B, dtype=float).ravel() for B in basis])
    b = S.ravel()
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(np.linalg.norm(A @ c - b) / np.linalg.norm(b))
