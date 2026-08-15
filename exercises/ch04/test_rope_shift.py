import numpy as np
from exercises.ch04.solution import apply_rope


def test_relative_only():
    rng = np.random.default_rng(9)
    d_h, base = 64, 500_000
    theta = base ** (-2 * np.arange(d_h // 2) / d_h)
    q, k = rng.normal(size=d_h), rng.normal(size=d_h)
    for m, n in ((3, 11), (100, 108), (5000, 5008)):
        lhs = apply_rope(q, m, theta) @ apply_rope(k, n, theta)
        rhs = q @ apply_rope(k, n - m, theta)
        assert abs(lhs - rhs) < 1e-6, "the logit must depend on n - m alone"
