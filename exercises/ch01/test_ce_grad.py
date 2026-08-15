import numpy as np
from exercises.ch01.solution import ce_grad


def test_grad_equals_p_minus_y():
    rng = np.random.default_rng(2)
    z = rng.normal(size=64).astype(np.float64)
    y = np.zeros(64); y[7] = 1.0
    g = ce_grad(z, y)
    def loss(zz):
        return -(y * (zz - np.log(np.exp(zz - zz.max()).sum()) - zz.max())).sum()
    num = np.zeros_like(z); eps = 1e-6
    for i in range(z.size):
        zp = z.copy(); zp[i] += eps
        zm = z.copy(); zm[i] -= eps
        num[i] = (loss(zp) - loss(zm)) / (2 * eps)
    assert np.abs(g - num).max() < 1e-6
