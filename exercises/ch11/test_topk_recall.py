"""E-11.14.  What a cheap indexer finds, measured rather than promised.

Unlike the other three files in this directory, nothing here is exact and
nothing here is a bound on output quality.  Top-k selection changes the
function being computed, and no theorem says the change is small.  What can be
measured is how much of the true top-k a surrogate recovers, and the useful
version of that measurement is a bracket:

  * a rank-r truncation of the true scores is the best rank-r approximation in
    Frobenius norm, so it stands in for the best a rank-r indexer could hope
    for.  On a trained two-block byte-level transformer it recalls 0.914 of the
    true top-8 at r = 4;
  * an untrained indexer of the same rank, projecting queries and keys through
    a random d_h x r map, recalls 0.425 on average over eight draws, ranging
    from 0.334 to 0.510.

The gap between those two is what fitting the indexer buys, and it is more than
a factor of two.  Neither number tells you what the output does, which is the
sentence the exercise asks you to write.

The committed scores are *unmasked* on purpose.  An indexer scores pairs and
the causal mask is applied afterwards, so truncating a matrix that already
carries -inf above the diagonal would spend the indexer's rank on representing
the mask, and would understate every recall figure here by about a fifth.
"""
import os

import numpy as np
import pytest

from exercises.ch11.solution import topk_recall

SCORES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "figs", "data", "ch11_scores.npz")


def _raw():
    """One trained layer's unmasked scores and the Q, K they came from,
    committed so the measurement reproduces without training anything."""
    if not os.path.exists(SCORES):
        pytest.skip("figs/data/ch11_scores.npz not present")
    z = np.load(SCORES)
    return (z["scores"].astype(np.float64), z["Q"].astype(np.float64),
            z["K"].astype(np.float64))


def _mask(s):
    return np.triu(np.full((s, s), -np.inf), 1)


def _true():
    raw, _, _ = _raw()
    return raw + _mask(raw.shape[1])


def _low_rank(r):
    """The best rank-r surrogate: no rank-r indexer beats this in Frobenius."""
    raw, _, _ = _raw()
    out = np.empty_like(raw)
    for i in range(raw.shape[0]):
        U, sv, Vt = np.linalg.svd(raw[i], full_matrices=False)
        out[i] = (U[:, :r] * sv[:r]) @ Vt[:r]
    return out + _mask(raw.shape[1])


def _narrow(r, seed=5):
    """An untrained indexer: project queries and keys through a random narrow
    map, which is what a lightning indexer of r dimensions looks like before
    anyone fits it."""
    raw, Q, K = _raw()
    rng = np.random.default_rng(seed)
    out = np.empty_like(raw)
    for i in range(raw.shape[0]):
        P = rng.normal(size=(Q.shape[2], r)) / np.sqrt(Q.shape[2])
        out[i] = (Q[i] @ P) @ (K[i] @ P).T / np.sqrt(r)
    return out + _mask(raw.shape[1])


def test_recall_monotone_in_k():
    true = _true()
    for r in (2, 4, 8):
        approx = _low_rank(r)
        rec = [topk_recall(true, approx, k) for k in (8, 16, 32, 64)]
        assert all(0.0 <= x <= 1.0 for x in rec), rec
        assert all(rec[i + 1] >= rec[i] - 1e-9 for i in range(len(rec) - 1)), \
            "rank %d: recall not monotone in k: %s" % (r, rec)


def test_a_perfect_surrogate_recalls_everything():
    """The calibration case.  If the surrogate is the true score matrix, recall
    is 1 at every k, and any implementation that scores masked or short rows
    incorrectly will fail here before it misleads anyone."""
    true = _true()
    for k in (8, 32, 64):
        assert abs(topk_recall(true, true.copy(), k) - 1.0) < 1e-12


def test_a_better_surrogate_recalls_more():
    """Recall must rise with the rank of the indexer.  It is the only
    monotonicity the statistic genuinely has, and it is worth pinning because
    an off-by-one in the masked-row handling breaks it."""
    true = _true()
    rec = [topk_recall(true, _low_rank(r), 16) for r in (2, 4, 8, 16)]
    assert all(rec[i + 1] > rec[i] for i in range(len(rec) - 1)), rec
    assert rec[-1] > 0.9


def test_the_bracket_between_a_fitted_and_an_unfitted_indexer():
    """The measurement the exercise is really about.  An untrained projection of
    the same rank recalls far less than the best possible surrogate of that
    rank, so the useful question about an indexer is not its rank but how well
    it was fitted.  Averaged over draws, because one random map is one sample
    and the spread across seeds is wide."""
    true = _true()
    best = topk_recall(true, _low_rank(4), 8)
    unfitted = [topk_recall(true, _narrow(4, seed), 8) for seed in range(8)]
    assert best > 0.9, best
    assert np.mean(unfitted) < 0.6, np.mean(unfitted)
    assert best > 2 * np.mean(unfitted), (best, np.mean(unfitted))
