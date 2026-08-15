import numpy as np
from arith.model_d import MODEL_D, rope_bands, critical_dimension
from exercises.ch04.solution import yarn_scaled_theta


def test_model_d_bands():
    assert critical_dimension(MODEL_D) == 35
    rows = rope_bands(MODEL_D)
    assert sum(r["gamma"] >= 1 - 1e-12 for r in rows) == 19
    assert sum(r["gamma"] <= 1e-12 for r in rows) == 29
    theta = MODEL_D.rope_base ** (-2 * np.arange(64) / MODEL_D.d_h)
    tp, g = yarn_scaled_theta(theta, s=16.0, L=MODEL_D.trained_context)
    assert np.allclose(g, [r["gamma"] for r in rows], atol=1e-12)
    assert np.allclose(theta[35:] / tp[35:], 16.0, atol=1e-9)
    assert np.allclose(theta[:19], tp[:19], atol=1e-12)
