import numpy as np
from exercises.ch04.solution import apply_rope, rope_complex


def test_fp32_agreement():
    rng = np.random.default_rng(10)
    d_h, base = 64, 500_000
    theta = base ** (-2 * np.arange(d_h // 2) / d_h)
    x = rng.normal(size=d_h).astype(np.float32)
    for m in (1, 137, 8192):
        assert np.allclose(apply_rope(x, m, theta), rope_complex(x, m, theta),
                           atol=1e-5), f"the two forms must agree at m={m}"
