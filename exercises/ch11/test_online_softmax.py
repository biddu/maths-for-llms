"""E-11.11.  An exact algorithm in an inexact arithmetic.

D-11.4 proves that the blocked recurrence returns exactly the unblocked
result, for every block count and every partition.  These tests do not use
`array_equal`, and the reason is worth stating precisely because it is the
distinction the chapter is built around.

The *algorithm* is exact: in real arithmetic the rescaling factors cancel and
the identity holds.  The *implementation* is in floating point, where each
rescaling rounds, so the output differs from an unblocked reference in the last
few units in the last place.  Measured on s = 4096 over partitions from one
block to 512: about 1.3e-15 in fp64, 8.3e-7 in fp32, 2.2e-3 in fp16, and never
bitwise equal in any of them.

That is not the same claim as "approximate attention".  An approximation
computes a different function on purpose and owes you a quality argument; this
computes the same function and owes you a tolerance.  Section 11.8 is where the
other kind lives.
"""
import numpy as np
import pytest

from exercises.ch11.solution import online_softmax_attention


def reference(z, v):
    """The unblocked computation, max-subtracted for stability."""
    w = np.exp(z - z.max())
    return (w[:, None] * v).sum(0) / w.sum()


def _case(s=4096, d_h=128, scale=3.0, seed=11):
    rng = np.random.default_rng(seed)
    return rng.normal(scale=scale, size=s), rng.normal(size=(s, d_h))


def test_exact_across_block_sizes():
    z, v = _case()
    ref = reference(z, v)
    for n_blocks in (1, 2, 4, 8, 64, 512):
        out = online_softmax_attention(z, v, n_blocks)
        assert out.shape == ref.shape
        assert np.allclose(out, ref, atol=1e-6, rtol=1e-5), \
            "%d blocks: max deviation %.3e" % (n_blocks, np.abs(out - ref).max())


def test_the_deviation_is_rounding_and_not_error():
    """Two statements, and the pair is the point.

    In float64 the deviation is at the level of accumulated rounding, far below
    any tolerance an application would care about.  And it is not zero: an
    assertion of bitwise equality fails, and should.
    """
    z, v = _case()
    ref = reference(z, v)
    out = online_softmax_attention(z, v, 64)
    assert np.abs(out - ref).max() < 1e-12
    assert not np.array_equal(out, ref), \
        "bitwise equality would mean the blocks were not actually rescaled"


def test_a_ragged_partition_is_still_exact():
    """D-11.4 assumes nothing about the blocks beyond their being a partition,
    so blocks of wildly different sizes must give the same answer.  A block
    count that does not divide s is the case every real kernel hits at the
    ragged tail."""
    z, v = _case(s=1000)
    ref = reference(z, v)
    for n_blocks in (3, 7, 37, 999):
        out = online_softmax_attention(z, v, n_blocks)
        assert np.allclose(out, ref, atol=1e-6, rtol=1e-5), n_blocks


def test_masked_positions_contribute_nothing():
    """A causal mask is applied by setting z_j = -inf before the recurrence
    runs.  Those positions must drop out entirely, which requires
    exp(-inf - m) to evaluate to zero rather than to NaN."""
    z, v = _case(s=512)
    keep = np.zeros(len(z), dtype=bool)
    keep[[3, 17, 400, 511]] = True
    z = np.where(keep, z, -np.inf)
    out = online_softmax_attention(z, v, 16)
    assert np.isfinite(out).all(), "masked blocks produced NaN"
    assert np.allclose(out, reference(z, v), atol=1e-8)
    # and the answer depends only on the kept positions
    zk, vk = z[keep], v[keep]
    assert np.allclose(out, reference(zk, vk), atol=1e-8)


@pytest.mark.parametrize("scale", [0.5, 3.0, 30.0])
def test_survives_scores_that_would_overflow(scale):
    """Without the running maximum, exp(z) overflows for large scores.  The
    recurrence carries m precisely so that it does not, and the test sweeps a
    scale at which a naive implementation returns NaN."""
    z, v = _case(s=1024, scale=scale, seed=5)
    z = z + 700.0                       # exp(700) is at the edge of float64
    out = online_softmax_attention(z, v, 8)
    assert np.isfinite(out).all()
    assert np.allclose(out, reference(z, v), atol=1e-6, rtol=1e-5)
