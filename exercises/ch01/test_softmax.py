import numpy as np, pytest
from exercises.ch01.solution import stable_softmax


def test_stable_softmax_matches_reference_at_1e4_logits():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(4, 32))
    p = stable_softmax(z)
    ref = np.exp(z - z.max(-1, keepdims=True))
    ref /= ref.sum(-1, keepdims=True)
    assert np.allclose(p, ref, atol=1e-12)
    assert np.allclose(p.sum(-1), 1.0, atol=1e-12)
    big = z + 1e4                       # the naive form overflows here
    assert np.isfinite(stable_softmax(big)).all()
    assert np.allclose(stable_softmax(big), p, atol=1e-12), "shift invariance"
