"""E-14.12.  Three scorings, three different winners.

D-14.2 proves the direction of the length bias and labels the correction a
tuned heuristic.  This file is what "tuned" means: on one three-candidate table
the unnormalised score, the mean-normalised score and the lp(n) penalty at the
recommended rho each rank a different candidate first, and the winner changes
twice inside the recommended band for rho.
"""
import numpy as np
import pytest

from exercises.ch14.solution import beam_scores

# two, four and six tokens; the long candidate is uniformly better per token
CANDIDATES = [
    [-0.34, -0.38],
    [-0.19, -0.21, -0.20, -0.22],
    [-0.15, -0.17, -0.16, -0.16, -0.16, -0.16],
]


def test_length_bias_reverses_ranking():
    s = beam_scores(CANDIDATES, rho=0.6)
    assert int(np.argmax(s["unnormalised"])) == 0, "shortest wins unnormalised"
    assert int(np.argmax(s["mean"])) == 2, "longest wins mean-normalised"
    assert int(np.argmax(s["lp"])) == 1, "the middle wins at rho = 0.6"


def test_scores_are_the_stated_formulas():
    s = beam_scores(CANDIDATES, rho=0.6)
    for i, c in enumerate(CANDIDATES):
        n = len(c)
        assert s["unnormalised"][i] == pytest.approx(sum(c))
        assert s["mean"][i] == pytest.approx(sum(c) / n)
        assert s["lp"][i] == pytest.approx(sum(c) / ((5 + n) / 6) ** 0.6)


def test_the_winner_changes_twice_across_rho():
    """Both switches lie inside the 0.6 to 1.0 band the literature recommends,
    so the recommendation does not determine the answer."""
    winners = []
    for rho in np.arange(0.0, 1.501, 0.005):
        w = int(np.argmax(beam_scores(CANDIDATES, rho=float(rho))["lp"]))
        if not winners or w != winners[-1][1]:
            winners.append((float(rho), w))
    assert [w for _, w in winners] == [0, 1, 2]
    assert winners[1][0] == pytest.approx(0.52, abs=0.01)
    assert winners[2][0] == pytest.approx(0.79, abs=0.01)


def test_every_extension_scores_below_its_prefix():
    """D-14.2 step 3: the unnormalised score is strictly decreasing in n."""
    c = CANDIDATES[2]
    part = [beam_scores([c[:n]])["unnormalised"][0] for n in range(1, len(c) + 1)]
    assert all(b < a for a, b in zip(part, part[1:]))
