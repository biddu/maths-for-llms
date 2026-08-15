import time
import numpy as np
from exercises.ch03.solution import linear_attention


def test_linear_scaling_and_gap():
    rng = np.random.default_rng(8)
    d_h, ts = 32, []
    for s in (512, 1024, 2048, 4096):
        q, k, v = (rng.normal(size=(s, d_h)) for _ in range(3))
        t0 = time.perf_counter(); linear_attention(q, k, v); ts.append(time.perf_counter() - t0)
    lg = np.polyfit(np.log([512, 1024, 2048, 4096]), np.log(ts), 1)[0]
    assert lg < 1.3, f"cost should grow about linearly in s, got exponent {lg:.2f}"
