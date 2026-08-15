import numpy as np
from exercises.ch05.solution import rho_per_layer


def test_mean_is_small():
    rng = np.random.default_rng(12)
    L, n, d = 32, 128, 4096
    acts = [rng.normal(size=(n, d)) + 0.01 * rng.normal(size=(n, 1)) for _ in range(L)]
    rho = rho_per_layer(acts)
    assert len(rho) == L
    assert all(np.median(r) < 0.05 for r in rho), \
        "if this fails on your checkpoint, that layer uses the all-ones direction"
