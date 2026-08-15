"""E-16.12.  Feature splitting, computed, and what it cannot settle."""
import numpy as np
import pytest

from exercises.ch16.solution import random_dictionary, splitting_fraction


def test_splitting_fraction():
    """A width-4m dictionary built by perturbing each width-m atom four ways
    must show near-total splitting; an independent one must show almost none.
    The contrast is the exercise: the statistic measures the relationship
    between two dictionaries, not a property of the model."""
    rng = np.random.default_rng(1612)
    d = 256
    U = random_dictionary(200, d, rng)
    # 0.25/sqrt(d) per coordinate, so the PERTURBATION has norm 0.25 against a
    # unit atom.  It was 0.25 per coordinate, which is a perturbation of norm
    # 0.25*sqrt(256) = 4.0: sixteen times the atom it was meant to nudge.  Each
    # child then sat at cos 0.24 from its own parent, never reaching the 0.7 the
    # statistic needs, and no threshold existed that passed both halves of this
    # test at once.  At norm 0.25 a child sits at cos 0.97 from its parent and
    # the intended contrast, 1.000 against 0.000, appears.
    child = np.repeat(U, 4, axis=0) + (0.25 / np.sqrt(d)) * rng.standard_normal((800, d))
    child /= np.linalg.norm(child, axis=1, keepdims=True)
    assert splitting_fraction(U, child) > 0.9
    indep = random_dictionary(800, d, rng)
    assert splitting_fraction(U, indep) < 0.02


def test_a_dictionary_splits_against_itself_trivially():
    rng = np.random.default_rng(2)
    U = random_dictionary(150, 128, rng)
    assert splitting_fraction(U, U) == 0.0
    assert splitting_fraction(U, np.vstack([U, U])) == pytest.approx(1.0)
