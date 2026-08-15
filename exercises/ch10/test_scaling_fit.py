"""E-10.10.  What a five-parameter fit does and does not recover.

The run table is synthetic, so unlike every published sweep this one has a
known true answer: the coefficients of `figs/data/make_scaling_runs.py`, which
are the book's frozen set.  That is the whole reason it exists.  You can check
not only that the fit converges but what it converged *to*.

Two separate fragilities are at issue and §10.1 keeps them apart:

  * the optimiser's.  The objective has a narrow basin and most random starts
    never find it.  That is a fact about your solver, and restarts measure it.
  * the data's.  Refit the same design under a fresh draw of the same 0.4%
    run-to-run noise and the exponents move by about 0.03 while the
    coefficients move by a factor of three.  That is a fact about the problem,
    and no amount of optimiser care removes it.

This file tests the first kind of claim on the committed table.  The second is
measured by `test_coefficients_are_the_loose_ones`.
"""
import csv
import functools
import os

import numpy as np

from exercises.ch10.solution import fit_scaling_law

TRUTH = {"L_inf": 1.82, "A": 482.0, "alpha": 0.348, "B": 2085.4, "beta": 0.366}
CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "figs", "data", "scaling_runs.csv")


@functools.lru_cache(maxsize=1)
def _fit():
    """Two hundred restarts is not cheap, and three tests want the same
    ensemble.  Fit once."""
    N, D, L = _table()
    return fit_scaling_law(N, D, L, n_starts=200, seed=0)


def _table():
    with open(CSV) as fh:
        rows = list(csv.DictReader(fh))
    return (np.array([float(r["N"]) for r in rows]),
            np.array([float(r["D"]) for r in rows]),
            np.array([float(r["loss"]) for r in rows]))


def test_recovers_exponents():
    N, D, L = _table()
    out = _fit()
    best = out["best"]

    # the exponents come back, which is what E-10.10 asks for
    assert abs(best["alpha"] - TRUTH["alpha"]) < 0.02, best["alpha"]
    assert abs(best["beta"] - TRUTH["beta"]) < 0.02, best["beta"]

    # and the fitted law reproduces the table it was fitted to
    pred = best["L_inf"] + best["A"] * N ** -best["alpha"] \
        + best["B"] * D ** -best["beta"]
    assert np.abs(pred / L - 1).max() < 0.01

    # the ensemble is returned whole, and "best" is the best of it
    assert len(out["all_fits"]) == 200
    assert out["objective"] <= min(f["objective"] for f in out["all_fits"]) + 1e-12


def test_coefficients_are_the_loose_ones():
    """§10.1's claim, stated as an inequality rather than as two numbers.

    Both exponents are recovered an order of magnitude better than either
    coefficient.  This is not the optimiser giving up early: it is the
    parametrisation.  A is multiplied by N^-alpha, so an error in alpha is
    absorbed almost exactly by a compensating error in A over the fitted range,
    and only the exponent survives the extrapolation the fit is wanted for.
    """
    best = _fit()["best"]
    rel = {k: abs(best[k] - TRUTH[k]) / TRUTH[k] for k in TRUTH}
    assert rel["alpha"] < 0.05 and rel["beta"] < 0.05
    assert rel["B"] > 5 * rel["beta"], rel
    assert rel["A"] > rel["alpha"], rel


def test_restarts_do_not_all_converge():
    """The basin is narrow.  Any honest report of a scaling fit says how many
    starts found it, because the number is not 200 out of 200 and quoting the
    winner alone hides that.  The assertion is deliberately weak: a better
    initialisation than the one this exercise suggests will raise the count,
    and that is a legitimate solution, not a failure."""
    out = _fit()
    objs = np.array([f["objective"] for f in out["all_fits"]])
    assert np.isfinite(objs).all()
    assert (objs <= out["objective"] * 1.01).sum() >= 5
