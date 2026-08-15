"""E-15.14.  How much a sequence-level ratio actually stabilises.

The blueprint asks for a variance ratio above |y|/4.  The law is exactly |y|
when every token may flip and exactly |y|^2 when only one may, so this file
asserts the laws.  A bound a factor of four away from the truth teaches less
than the truth does.

(The arithmetic of E-15.7 is a claim about the book, not about the reader's
code, so it lives in exercises/test_arithmetic.py.  Every test in this
directory must fail on a fresh clone, and one that never calls solution.py
would not.)
"""
import numpy as np
import pytest

from exercises.ch15.solution import ratio_variances

N = 200_000


@pytest.mark.parametrize("L", [32, 128, 512])
def test_sequence_ratio_is_stabler(L):
    rng = np.random.default_rng(1514 + L)
    vt, vs = ratio_variances(N, L, 0.01, 4.0, rng, all_tokens=True)
    assert vt / vs == pytest.approx(L, rel=0.06)
    assert vt / vs > L / 4          # the blueprint's bound, comfortably


@pytest.mark.parametrize("L", [32, 128])
def test_a_single_flipping_token_gives_the_square(L):
    rng = np.random.default_rng(99 + L)
    vt, vs = ratio_variances(N, L, 0.01, 4.0, rng, all_tokens=False)
    assert vt / vs == pytest.approx(L * L, rel=1e-6)
