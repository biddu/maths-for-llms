import numpy as np
from exercises.ch05.solution import rmsnorm_forward, rmsnorm_backward


def test_radial_gradient_is_zero():
    rng = np.random.default_rng(13)
    d = 64
    x = rng.normal(size=d) * 2.5
    g = rng.normal(size=d) * 0.5 + 1.0
    dy = rng.normal(size=d)
    dx, dg = rmsnorm_backward(x, g, dy)
    assert dx.shape == x.shape and dg.shape == g.shape
    # D-5.1 step 7: the radial component is annihilated
    assert abs(x @ dx) < 1e-7 * np.linalg.norm(x) * np.linalg.norm(dx) + 1e-9
    # and the forward map is scale-invariant
    assert np.allclose(rmsnorm_forward(x, g), rmsnorm_forward(3.7 * x, g), atol=1e-6)
