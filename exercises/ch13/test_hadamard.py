"""E-13.11.  The rotation is exact, cheap, and useless without its signs.

Three claims, and the third is the one the blueprint's parenthetical hides:
a fixed Hadamard is derandomised, not random, and on an adversarial input it
accomplishes nothing at all.
"""
import numpy as np
import pytest

from exercises.ch13.solution import fwht, randomised_hadamard

D = 4096


def _spiked(seed=7, n_out=12, spike=20.0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(D)
    x[rng.choice(D, n_out, replace=False)] *= spike
    return x


def _incoherence(v):
    return np.sqrt(len(v)) * np.abs(v).max() / np.linalg.norm(v)


def test_hadamard_invariance_and_incoherence():
    rng = np.random.default_rng(11)
    W = rng.standard_normal((D, 128)) / np.sqrt(D)
    Q = randomised_hadamard(D, seed=3)
    assert Q.shape == (D, D)
    assert np.abs(Q @ Q.T - np.eye(D)).max() < 1e-10, "Q is not orthogonal"

    x = _spiked()
    ref = x @ W
    got = (x @ Q) @ (Q.T @ W)
    # fp64 is exact to rounding
    assert np.abs(ref - got).max() < 1e-9
    # fp16 is not, and the tolerance has to be relative: the absolute error is
    # about 2e-3 at this width, which is arithmetic and not a bug
    h = lambda a: a.astype(np.float16)
    got16 = (h(h(x) @ h(Q)) @ h(Q.T @ W)).astype(np.float64)
    rel = np.abs(ref - got16).max() / np.abs(ref).max()
    assert rel < 1e-3, "relative deviation in fp16: %.3e" % rel

    # and the rotation does what D-13.2 says
    before, after = _incoherence(x), _incoherence(x @ Q)
    assert before > 15.0
    assert after < 2.0 * np.sqrt(2 * np.log(D))
    assert before / after > 4.0


def test_the_transform_agrees_with_the_matrix():
    """fwht must be the Sylvester Hadamard, not merely some orthogonal map."""
    rng = np.random.default_rng(2)
    for d in (8, 64, 1024):
        H = np.array([[1.0]])
        while H.shape[0] < d:
            H = np.block([[H, H], [H, -H]])
        x = rng.standard_normal(d)
        assert np.abs(fwht(x) - H @ x).max() < 1e-8, d


def test_a_fixed_hadamard_fails_on_an_aligned_vector():
    """The reason implementations carry a sign vector.  A vector equal to a row
    of H is mapped to a one-hot, so its incoherence stays at the maximum."""
    H = np.array([[1.0]])
    while H.shape[0] < D:
        H = np.block([[H, H], [H, -H]])
    row = H[7] / np.sqrt(D)
    fixed = H / np.sqrt(D)
    assert _incoherence(row @ fixed) > 0.9 * np.sqrt(D), \
        "a fixed Hadamard should leave an aligned vector maximally spiked"
    worst = max(_incoherence(row @ randomised_hadamard(D, seed=s))
                for s in range(12))
    assert worst < 10.0, worst


@pytest.mark.parametrize("d", [1024, 4096])
def test_the_predicted_gain(d):
    """The ratio of maxima for a one-hot spike is sqrt(d / 2 ln d), which is the
    worst case and therefore the bound the chapter prints."""
    x = np.zeros(d)
    x[3] = 1.0
    Q = randomised_hadamard(d, seed=1)
    ratio = np.abs(x).max() / np.abs(x @ Q).max()
    assert ratio >= np.sqrt(d / (2 * np.log(d))) * 0.9, ratio
