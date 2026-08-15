"""E-15.12.  Nothing beats pi*, and the Gibbs identity is an identity.

D-15.3 is the load-bearing result of the chapter, and it is not an
approximation: (15.6) holds to machine precision for every policy, not merely
near the optimum.  This file asserts the identity itself, which is stronger and
more useful than asserting that an optimiser finds the right answer.
"""
import numpy as np
import pytest

from exercises.ch15.solution import log_Z, objective, pi_star

V = 64


def _problem(seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(V)
    z = z - z.max()
    pref = np.exp(z) / np.exp(z).sum()
    return pref, rng.standard_normal(V) * 2.0, float(rng.uniform(0.2, 2.0))


def test_no_policy_beats_pi_star():
    pref, r, beta = _problem(1512)
    star = pi_star(pref, r, beta)
    Jstar = objective(star, pref, r, beta)
    assert Jstar == pytest.approx(beta * log_Z(pref, r, beta), abs=1e-10)
    rng = np.random.default_rng(7)
    best = -np.inf
    for _ in range(100_000):
        pi = rng.dirichlet(np.full(V, 0.7))
        best = max(best, objective(pi, pref, r, beta))
    assert best < Jstar
    assert Jstar - best > 0.5          # and not by a whisker


def test_gibbs_identity_holds_for_every_policy():
    """(15.6): J(pi) = -beta KL(pi || pi*) + beta log Z, exactly."""
    rng = np.random.default_rng(3)
    for seed in range(40):
        pref, r, beta = _problem(seed)
        star = pi_star(pref, r, beta)
        lz = log_Z(pref, r, beta)
        for _ in range(5):
            pi = rng.dirichlet(np.full(V, rng.uniform(0.4, 3.0)))
            kl = float((pi * np.log(pi / star)).sum())
            assert objective(pi, pref, r, beta) == pytest.approx(
                -beta * kl + beta * lz, abs=1e-9)


def test_free_energy_limits():
    """D-15.3 step 8: beta log Z interpolates max r and E_ref[r].

    The small-beta limit has a correction worth knowing:
        beta log Z -> max(r) + beta log pi_ref[argmax r],
    because the largest term dominates the sum and carries its own reference
    probability with it.  Asserting max(r) alone needs beta below 1e-5 before
    it holds to three decimals, which is why the exact form is asserted here.
    """
    pref, r, _ = _problem(11)
    j = int(np.argmax(r))
    for beta in (1e-2, 1e-3, 1e-4):
        assert beta * log_Z(pref, r, beta) == pytest.approx(
            r.max() + beta * np.log(pref[j]), abs=1e-6)
    assert 5000.0 * log_Z(pref, r, 5000.0) == pytest.approx(
        float((pref * r).sum()), abs=1e-2)


def test_optimum_respects_the_reference_support():
    """D-15.3's absolute-continuity clause, which Corollary 15.1 pays off."""
    pref = np.array([0.5, 0.5, 0.0])
    r = np.array([0.0, 0.0, 10.0])
    for beta in (0.05, 0.5, 5.0):
        assert pi_star(pref, r, beta)[2] == 0.0
