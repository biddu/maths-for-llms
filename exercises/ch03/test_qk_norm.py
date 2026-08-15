import numpy as np
from exercises.ch03.solution import qk_norm_logits


def test_logit_variance_bounded():
    rng = np.random.default_rng(7)
    d_h = 128
    q = rng.normal(scale=2.5, size=(4096, d_h))     # entry variance has drifted
    k = rng.normal(scale=2.5, size=(4096, d_h))
    raw = np.einsum("ij,ij->i", q, k) / np.sqrt(d_h)
    assert raw.var() > 1.0, "unnormalised logits exceed unit variance"
    z = qk_norm_logits(q, k)
    assert np.abs(z).max() <= d_h + 1e-6, "sqrt(d_h) cos theta is bounded by d_h"
