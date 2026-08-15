"""E-13.10.  The bound of D-13.1, checked elementwise.

The bound is |x - xhat| <= s_q/2, and it holds only where the clamp is
inactive.  Group-wise quantisation makes that almost everywhere, because each
group's scale is sized from its own extremes, so nothing inside a group is ever
clipped.  That is the whole reason group scales exist.
"""
import numpy as np
import pytest

from exercises.ch13.solution import affine_quantise


def _weights(n=8192, seed=131):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    x[rng.choice(n, 12, replace=False)] *= 20.0      # a few outliers
    return x


@pytest.mark.parametrize("b_q,g_q", [(4, 64), (4, 128), (8, 128), (3, 32)])
def test_affine_roundtrip_error_bound(b_q, g_q):
    x = _weights()
    xhat, s_q = affine_quantise(x, b_q, g_q)
    assert xhat.shape == x.shape
    assert s_q.shape == (int(np.ceil(len(x) / g_q)),)
    # the scale is sized from each group's own range, so nothing is clipped and
    # the bound holds everywhere, not merely on average
    per_group = np.repeat(s_q, g_q)[:len(x)]
    assert np.all(np.abs(x - xhat) <= per_group / 2 + 1e-9)


def test_smaller_groups_are_more_accurate():
    """The trade (13.5) prices: a smaller group spans a smaller range, so its
    scale is finer.  Monotone, and it is what buys the extra bits back."""
    x = _weights()
    mse = [float(((x - affine_quantise(x, 4, g)[0]) ** 2).mean())
           for g in (2048, 512, 128, 32)]
    assert all(mse[i + 1] < mse[i] for i in range(len(mse) - 1)), mse


def test_a_bit_is_worth_about_six_decibels():
    """D-13.1 step 8, on data with no outliers so the loading factor is fixed.
    Each extra bit should buy close to 6.02 dB."""
    rng = np.random.default_rng(5)
    x = np.clip(rng.standard_normal(200_000), -4, 4)
    snr = []
    for b in (4, 5, 6, 7, 8):
        xhat, _ = affine_quantise(x, b, 4096)
        snr.append(10 * np.log10(float((x ** 2).mean() / ((x - xhat) ** 2).mean())))
    steps = np.diff(snr)
    assert np.all(np.abs(steps - 6.02) < 0.4), steps


def test_the_last_group_is_handled():
    """A length that is not a multiple of the group size is the common case and
    the easy thing to get wrong."""
    x = _weights(n=1000)
    xhat, s_q = affine_quantise(x, 4, 128)
    assert len(s_q) == 8
    assert np.all(np.isfinite(xhat))
    per_group = np.repeat(s_q, 128)[:len(x)]
    assert np.all(np.abs(x - xhat) <= per_group / 2 + 1e-9)


def test_an_exactly_representable_value_is_fixed():
    """Dequantise a value already on the grid and it must not move.  Without
    this, an off-by-one in the zero-point passes every statistical test."""
    x = np.linspace(-1, 1, 512)
    xhat, s_q = affine_quantise(x, 8, 512)
    on_grid = xhat.copy()
    again, _ = affine_quantise(on_grid, 8, 512)
    assert np.abs(again - on_grid).max() < 1e-9
