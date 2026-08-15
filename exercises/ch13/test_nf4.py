"""E-13.13.  NF4's levels, and two different meanings of optimal.

Lloyd-Max minimises squared error for one distribution and has to be fitted to
it.  NF4 puts equal probability mass in every bin, which is the entropy-optimal
choice, and gives a fixed table that no kernel has to fit or look up per block.
The measured ordering is int4 worst, NF4 in between, Lloyd-Max best, and the
exercise is to say why the middle one is what ships.
"""
import numpy as np

from exercises.ch13.solution import nf4_levels


def _quantise_to(levels, x):
    return levels[np.abs(x[:, None] - levels[None, :]).argmin(1)]


def _lloyd_max(x, k=16, iters=150):
    lv = np.quantile(x, np.linspace(0.01, 0.99, k))
    for _ in range(iters):
        idx = np.abs(x[:, None] - lv[None, :]).argmin(1)
        for j in range(k):
            m = idx == j
            if m.any():
                lv[j] = x[m].mean()
    return np.sort(lv)


def test_nf4_levels_and_mse():
    lv = nf4_levels()
    assert lv.shape == (16,)
    assert np.all(np.diff(lv) > 0), "levels must be sorted and distinct"
    assert abs(lv.min() + 1.0) < 1e-12 and abs(lv.max() - 1.0) < 1e-12
    assert np.any(lv == 0.0), "a format with no exact zero cannot store a pruned weight"

    rng = np.random.default_rng(53)
    x = rng.standard_normal(200_000)
    xn = x / np.abs(x).max()                    # absmax-normalised, as NF4 assumes
    mse = {}
    mse["int4"] = float(((xn - _quantise_to(np.linspace(-1, 1, 16), xn)) ** 2).mean())
    mse["nf4"] = float(((xn - _quantise_to(lv, xn)) ** 2).mean())
    mse["lloyd"] = float(((xn - _quantise_to(_lloyd_max(xn[:20000]), xn)) ** 2).mean())
    assert mse["lloyd"] < mse["nf4"] < mse["int4"], mse
    assert mse["nf4"] < 0.6 * mse["int4"], mse


def test_the_levels_crowd_where_the_mass_is():
    """The structural property, independent of any particular sample: the gaps
    are narrow near zero and wide at the extremes, which is what equalising
    probability mass means."""
    lv = nf4_levels()
    gaps = np.diff(lv)
    mid = len(gaps) // 2
    assert gaps[mid] < gaps[0] and gaps[mid] < gaps[-1]
    assert gaps.max() / gaps.min() > 2.5


def test_it_is_symmetric_in_shape_but_not_in_count():
    """Sixteen levels cannot be symmetric about an included zero: one side gets
    eight and the other seven.  Knowing which is which is the difference
    between reproducing the format and approximating it."""
    lv = nf4_levels()
    assert (lv < 0).sum() + (lv > 0).sum() == 15
    assert (lv < 0).sum() == 7 or (lv > 0).sum() == 7
