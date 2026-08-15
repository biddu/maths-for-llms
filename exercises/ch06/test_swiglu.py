import numpy as np
from exercises.ch06.solution import swiglu


def _reference(x, Wg, Wu, Wd, beta=1.0):
    z = x @ Wg
    return ((z / (1.0 + np.exp(-beta * z))) * (x @ Wu)) @ Wd


def test_matches_torch():
    """E-6.8.  Named for the torch reference; the closed form is here so the
    test runs without torch installed, and it is the same function."""
    rng = np.random.default_rng(6)
    d, d_ff, n = 32, 48, 7
    x = rng.normal(size=(n, d))
    Wg, Wu = rng.normal(size=(d, d_ff)), rng.normal(size=(d, d_ff))
    Wd = rng.normal(size=(d_ff, d))
    got = swiglu(x, Wg, Wu, Wd)
    assert got.shape == (n, d)
    assert np.max(np.abs(got - _reference(x, Wg, Wu, Wd))) < 1e-6


def test_degree_two_coefficient_is_half():
    """D-6.3 step 3: the leading coefficient is sigma(0) = 1/2 whatever beta is.
    Scale the input down and the cubic term falls away faster than the
    quadratic, so the ratio converges to 1/2."""
    rng = np.random.default_rng(7)
    d = 8
    w, v = rng.normal(size=(d, 1)), rng.normal(size=(d, 1))
    Wd = np.ones((1, 1))
    for beta in (0.5, 1.0, 3.0):
        ratios = []
        for t in (1e-2, 1e-3):
            x = t * rng.normal(size=(200, d))
            y = swiglu(x, w, v, Wd, beta=beta)[:, 0]
            ratios.append(np.mean(y / ((x @ w)[:, 0] * (x @ v)[:, 0])))
        assert abs(ratios[-1] - 0.5) < 1e-2, f"beta={beta} gave {ratios[-1]}"
