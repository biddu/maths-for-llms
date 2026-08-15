"""E-9.13.  Equation (9.21), checked against simulated Adam."""
import numpy as np
from exercises.ch09.solution import peak_update_after_burst


def test_peak_update_matches_closed_form():
    b1 = 0.9
    for b2 in (0.9, 0.95, 0.99, 0.995, 0.999):
        got = peak_update_after_burst(b1, b2)
        want = (1 - b1) / np.sqrt(1 - b2)
        assert abs(got - want) < 1e-3, "beta2 %.3f: got %.4f, want %.4f" % (b2, got, want)


def test_the_peak_does_not_depend_on_how_big_the_burst_was():
    a = peak_update_after_burst(0.9, 0.999, g_burst=1.0)
    b = peak_update_after_burst(0.9, 0.999, g_burst=1e4)
    assert abs(a - b) < 1e-3


def test_beta2_095_cannot_overshoot():
    """0.447 < 1: at beta2 = 0.95 no single gradient can take a step larger
    than the nominal one, which is the design consequence in §9.9."""
    assert peak_update_after_burst(0.9, 0.95) < 1.0
    assert peak_update_after_burst(0.9, 0.999) > 3.0
