"""E-10.12.  Serving never argues for a larger model.

D-10.3 adds one term to the objective, 2 N D_inf, and the whole chapter's
conclusion is the sign of one derivative.  A sign is exactly the kind of claim
that survives being wrong in a paper for years, so it is worth pinning
numerically as well as deriving.
"""
import numpy as np

from arith.model_d import REFIT_2024
from arith.scaling_budget import optimal_D, loss
from exercises.ch10.solution import inference_aware_optimum

TARGET = 2.20          # a loss reachable at every model size considered here
SERVED = [0.0, 1e12, 1e13, 1e14]


def test_optimum_monotone():
    out = [inference_aware_optimum(TARGET, d, REFIT_2024) for d in SERVED]
    N = [o["N"] for o in out]
    ratio = [o["tokens_per_param"] for o in out]

    assert all(N[i + 1] < N[i] for i in range(len(N) - 1)), N
    assert all(ratio[i + 1] > ratio[i] for i in range(len(ratio) - 1)), ratio

    # every one of them hits the loss target it was given
    for o in out:
        assert abs(loss(o["N"], o["D"], REFIT_2024) - TARGET) < 1e-6


def test_collapses_to_compute_optimal():
    """At D_inf = 0 the inference term vanishes and (10.8) must become (10.4).
    This is the check that the generalisation is a generalisation."""
    o = inference_aware_optimum(TARGET, 0.0, REFIT_2024)
    assert abs(o["D"] / optimal_D(o["N"], REFIT_2024) - 1) < 1e-4


def test_serving_moves_the_ratio_far():
    """The chapter's headline: the gap between compute-optimal and
    deployment-optimal is a factor of a hundred, not a few per cent."""
    base = inference_aware_optimum(TARGET, 0.0, REFIT_2024)
    heavy = inference_aware_optimum(TARGET, 1e14, REFIT_2024)
    assert base["tokens_per_param"] < 30
    assert heavy["tokens_per_param"] > 10 * base["tokens_per_param"]
    # and the lifetime cost of the heavy-serving optimum is below what the
    # compute-optimal model would have cost over the same serving life
    alt = 6 * base["N"] * base["D"] + 2 * base["N"] * 1e14
    assert heavy["lifetime_flops"] < alt


def test_lifetime_flops_is_the_thing_being_minimised():
    """Perturb N off the optimum in both directions at fixed loss and confirm
    lifetime compute rises.  Without this, a solver that returned the
    compute-optimal answer regardless of D_inf would pass the monotonicity
    test on a lucky bracketing."""
    D_inf = 1e13
    o = inference_aware_optimum(TARGET, D_inf, REFIT_2024)
    f = REFIT_2024

    def D_at(N):
        r = TARGET - f["L_inf"] - f["A"] * N ** -f["alpha"]
        return (f["B"] / r) ** (1.0 / f["beta"]) if r > 0 else np.inf

    here = 6 * o["N"] * o["D"] + 2 * o["N"] * D_inf
    for g in (0.9, 0.95, 1.05, 1.1):
        N = o["N"] * g
        cost = 6 * N * D_at(N) + 2 * N * D_inf
        assert cost > here, "factor %.2f gave %.4e vs %.4e" % (g, cost, here)
