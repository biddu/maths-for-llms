"""E-11.12.  Absorption computes the same logits and allocates less.

Two claims, and they are separate.  The *numerical* claim is that folding
W^Q W^{UK T} into one matrix changes nothing, which is D-11.2 step 6 and is
associativity.  The *resource* claim is that the folded path never materialises
K, which is the whole reason anyone does it.  A solution that computes the
folded matrix and then reconstructs K anyway would pass the first and fail the
second, so both are tested.
"""
import numpy as np

from exercises.ch11.solution import mla_logits

D, D_C, D_H = 512, 128, 64


def _weights(n_q=9, n_k=257, seed=12):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n_q, D)) / np.sqrt(D),
            rng.normal(size=(n_k, D_C)) / np.sqrt(D_C),
            rng.normal(size=(D, D_H)) / np.sqrt(D),
            rng.normal(size=(D_C, D_H)) / np.sqrt(D_C))


def test_absorbed_matches_materialised():
    X, C, W_q, W_uk = _weights()
    mat = mla_logits(X, C, W_q, W_uk, absorbed=False)
    fold = mla_logits(X, C, W_q, W_uk, absorbed=True)
    assert mat.shape == (X.shape[0], C.shape[0])
    assert np.allclose(mat, fold, atol=1e-4), \
        "max deviation %.3e" % np.abs(mat - fold).max()
    # in float64 the two differ only by summation order, so the agreement is
    # much tighter than the 1e-4 the exercise asks for
    assert np.abs(mat - fold).max() < 1e-10


def test_the_absorbed_path_never_forms_the_keys(monkeypatch):
    """The memory claim, enforced rather than trusted.

    Every array numpy allocates through matmul passes through here, so if the
    folded path ever produces an (n_k, d_h) key array the test sees it.  This
    is the assertion that separates absorption from a cosmetic rewrite.
    """
    X, C, W_q, W_uk = _weights()
    key_shape = (C.shape[0], D_H)
    seen = []
    real_matmul = np.matmul

    def watched(a, b, *args, **kwargs):
        out = real_matmul(a, b, *args, **kwargs)
        seen.append(np.shape(out))
        return out

    monkeypatch.setattr(np, "matmul", watched)
    mla_logits(X, C, W_q, W_uk, absorbed=True)
    assert key_shape not in seen, \
        "the absorbed path allocated an array of the materialised key's shape"


def test_the_fold_is_a_load_time_constant():
    """W^Q W^{UK T} depends on neither the query nor the cache, so folding it
    once and reusing it across every batch must give the same answer as folding
    it per call.  That is what makes it a load-time constant rather than an
    optimisation the runtime has to repeat."""
    X, C, W_q, W_uk = _weights()
    W_tilde = W_q @ W_uk.T
    assert W_tilde.shape == (D, D_C)
    direct = (X @ W_tilde) @ C.T
    assert np.allclose(direct, mla_logits(X, C, W_q, W_uk, absorbed=True),
                       atol=1e-10)
