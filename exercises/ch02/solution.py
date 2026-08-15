"""Your solutions for this chapter's [C] exercises.

THIS IS THE `solutions` BRANCH.  Every function below is worked in full and
every test in this directory passes.  On `main` these are stubs raising
NotImplementedError, and making them pass is the exercise; read this file only
after you have had a go, or when you want to compare a method rather than an
answer.  Appendix C prints the answer and the tolerance, not the code.
"""
import numpy as np

from arith.model_d import MODEL_D, Config, total_params


def embedding_backward(ids, dX, V):
    """E-2.9.  Scatter-add the upstream gradient into a V x d matrix.

    D-2.1 step 5: the lookup is O W_E, so the gradient is O^T dX, which is a
    sum over the positions where each id occurred.  `np.add.at` accumulates
    repeats; plain fancy indexing would overwrite them and silently drop every
    occurrence of a token but the last.
    """
    ids = np.asarray(ids)
    dX = np.asarray(dX, dtype=float)
    g = np.zeros((V, dX.shape[-1]), dtype=dX.dtype)
    np.add.at(g, ids, dX)
    return g


def centred_cosine(W):
    """E-2.10.  Cosine similarity after subtracting the row mean of W.

    The row mean is the mean row, one d-vector averaged over the vocabulary.
    Subtracting it removes the common component every row shares, which is the
    component that puts a floor under the raw cosine matrix.
    """
    W = np.asarray(W, dtype=float)
    C = W - W.mean(axis=0, keepdims=True)
    Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
    return Cn @ Cn.T


def all_but_the_top(W, k=1):
    """E-2.10 (extended).  Cosine after removing the mean and the top k
    principal directions.  Centring alone is necessary and not sufficient; see
    measure/README.md.

    Mu and Viswanath's recipe: centre, take the top k principal directions of
    the centred matrix, and project each row onto their orthogonal complement.
    The directions come from the SVD of the centred matrix, whose right
    singular vectors are the eigenvectors of the covariance.

    Pass the whole embedding matrix, not the handful of rows you mean to plot.
    The mean and the principal directions are properties of the cloud, and
    `measure/checkpoint_stats.py` estimates them on the full vocabulary before
    it takes the cosines of its forty sampled rows.  Centring a set of n rows
    and then measuring the same n rows pins the mean off-diagonal cosine at
    -1/(n-1) whatever the geometry was, because the centred rows sum to zero.
    """
    W = np.asarray(W, dtype=float)
    C = W - W.mean(axis=0, keepdims=True)
    if k > 0:
        # Right singular vectors of C are the principal directions of the cloud.
        U = np.linalg.svd(C, full_matrices=False)[2][:k]      # (k, d)
        C = C - (C @ U.T) @ U                                 # remove them
    Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
    return Cn @ Cn.T


def param_count(V, d, L, d_ff, tied):
    """E-2.11.  Reproduce both figures in the Chapter 2 arithmetic box.

    The box quotes Model D and the 1 B shape of `arith/small_model.py`, and
    neither passes an attention geometry here, so the geometry is the reference
    model's: head width d_h = 128, h = d / d_h query heads, and GQA in groups of
    four, so n_kv = h / 4.  Both are read off Model D rather than typed in, and
    the count itself is `arith.model_d.total_params`, so this cannot drift from
    the printed ledger.
    """
    d_h = MODEL_D.d_h
    group = MODEL_D.h // MODEL_D.n_kv          # query heads per KV head, 4
    h = d // d_h
    c = Config(L=L, d=d, h=h, d_h=d_h, n_kv=h // group,
               d_ff=d_ff, V=V, tied=tied)
    return total_params(c)
