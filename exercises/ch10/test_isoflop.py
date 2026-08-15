"""E-10.11.  The picture and the algebra have to agree.

Figure 10.1 is drawn by evaluating the law on a grid; equation (10.5) is the
closed form of the same minimum.  If the two disagree the derivation is wrong,
and a derivation that only ever gets checked against its own picture is not
checked at all.
"""
import numpy as np

from arith.model_d import REFIT_2024, CHINCHILLA
from arith.scaling_budget import optimal_N, loss
from exercises.ch10.solution import isoflop_minimum


def test_minimum_matches_closed_form():
    for fit in (REFIT_2024, CHINCHILLA):
        for C in (1e19, 1e20, 1e21, 1e22, 1e23, 1e24):
            n_grid = isoflop_minimum(C, fit)
            n_form = optimal_N(C, fit)
            assert abs(n_grid / n_form - 1) < 0.02, \
                "C = %.0e: grid %.4e, closed form %.4e" % (C, n_grid, n_form)


def test_it_really_is_a_minimum():
    """U-shaped, so the loss rises on both sides.  A grid argmin that lands on
    an endpoint would pass the test above whenever the closed form happened to
    sit near the edge of the grid; this is what rules that out."""
    fit = REFIT_2024
    for C in (1e20, 1e22, 1e24):
        n = isoflop_minimum(C, fit)
        here = loss(n, C / (6 * n), fit)
        for f in (0.5, 0.8, 1.25, 2.0):
            other = loss(n * f, C / (6 * n * f), fit)
            assert other > here, "C = %.0e, factor %.2f" % (C, f)


def test_the_frontier_has_the_predicted_slope():
    """D-10.2 step 6: N* scales as C^(beta/(alpha+beta)).  Measured on the grid
    rather than asserted from the formula, so this fails if `isoflop_minimum`
    is quietly returning the closed form instead of searching."""
    fit = REFIT_2024
    Cs = np.array([1e20, 1e21, 1e22, 1e23, 1e24])
    Ns = np.array([isoflop_minimum(C, fit) for C in Cs])
    slope = np.polyfit(np.log10(Cs), np.log10(Ns), 1)[0]
    expect = fit["beta"] / (fit["alpha"] + fit["beta"])
    assert abs(slope - expect) < 0.01, (slope, expect)
