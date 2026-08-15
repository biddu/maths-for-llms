"""E-3.14.  Linear attention is O(s), and the way you prove it is not a stopwatch.

This test used to be a stopwatch alone: four sizes, one `time.perf_counter()`
sample each, and a fitted exponent asserted below 1.3.  On an idle laptop that
passes.  On a shared CI runner, or on any machine doing something else at the
time, the samples are contaminated by the scheduler and the fit wanders: it
failed here at 1.4 while the book was being typeset in another process, and
passed twice in a row the moment the machine was quiet.

That matters more than a flaky test usually does.  The book's copyright page
stakes a claim on continuous integration, and Appendix C says a solution that
stops passing shows as "a red badge rather than an erratum someone emails in".
A red badge that means "the runner was busy" spends the credibility of every
other badge in the repository.

So the claim is checked twice, and the deterministic check is the one that
carries it:

  1. MEMORY, which is exact.  The whole point of equation (3.23) is that
     phi(K)^T V is a (d_h, d_v) state, so no s x s score matrix is ever formed.
     tracemalloc measures peak allocation and does not care what else the
     machine is doing.
  2. TIME, kept, because an implementation could stream the s x s matrix in
     blocks and stay inside the memory bound while still doing O(s^2) work.
     Made robust: the minimum of several repeats rather than a single sample,
     since the minimum is the run least contaminated by the scheduler, and a
     threshold of 1.45, which still separates linear from quadratic decisively.
"""
import time
import tracemalloc

import numpy as np

from exercises.ch03.solution import linear_attention

SIZES = (512, 1024, 2048, 4096)
REPEATS = 5


def test_no_s_by_s_matrix_is_ever_formed():
    """The deterministic half.  An s x s float64 matrix at s = 4096 is 134 MB."""
    rng = np.random.default_rng(8)
    s, d_h = 4096, 32
    q, k, v = (rng.normal(size=(s, d_h)) for _ in range(3))

    linear_attention(q, k, v)          # warm up, so import-time allocation is out
    tracemalloc.start()
    linear_attention(q, k, v)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    quadratic = s * s * 8
    assert peak < quadratic / 10, (
        f"peak allocation {peak / 1e6:.1f} MB; an s x s score matrix would be "
        f"{quadratic / 1e6:.1f} MB, so this implementation is forming one")


def test_cost_grows_about_linearly_in_s():
    """The timing half.  Minimum of {} repeats, so a busy runner cannot fail it.""".format(REPEATS)
    rng = np.random.default_rng(8)
    d_h, ts = 32, []
    for s in SIZES:
        q, k, v = (rng.normal(size=(s, d_h)) for _ in range(3))
        best = min(_timed(linear_attention, q, k, v) for _ in range(REPEATS))
        ts.append(best)
    lg = np.polyfit(np.log(SIZES), np.log(ts), 1)[0]
    assert lg < 1.45, f"cost should grow about linearly in s, got exponent {lg:.2f}"


def _timed(fn, *args):
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0
