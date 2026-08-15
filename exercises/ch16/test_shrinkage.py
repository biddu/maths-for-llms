"""E-16.9.  The L1 bias is exactly lambda/2, and it does not depend on k."""
import numpy as np
import pytest

from exercises.ch16.solution import reconstruction_ratio, soft_threshold, topk

LAM = 0.4


def test_l1_bias_equals_half_lambda():
    rng = np.random.default_rng(1609)
    c = rng.standard_normal(200_000)
    z = soft_threshold(c, LAM)
    act = z != 0
    assert np.all(np.abs(c[~act]) <= LAM / 2 + 1e-12)
    assert (np.abs(c[act]) - np.abs(z[act])).mean() == pytest.approx(LAM / 2,
                                                                    abs=1e-9)
    assert np.all(np.sign(z[act]) == np.sign(c[act]))


def test_topk_has_no_shrinkage():
    rng = np.random.default_rng(3)
    c = rng.standard_normal(4096)
    z = topk(c, 64)
    keep = z != 0
    assert keep.sum() == 64
    assert np.allclose(z[keep], c[keep])
    assert np.abs(c[keep]).min() >= np.abs(c[~keep]).max()


@pytest.mark.parametrize("k", [3, 10, 40, 200])
def test_reconstruction_deficit_is_independent_of_k(k):
    """(16.18).  One lambda does selection and shrinkage, and the deficit is a
    property of lambda alone."""
    for cbar in (0.5, 1.0, 2.0):
        r = reconstruction_ratio(np.full(k, cbar), LAM)
        assert r == pytest.approx(1 - LAM / (2 * cbar), abs=1e-12)
