"""Your solutions for Chapter 10's [C] exercises.

THIS IS THE `solutions` BRANCH.  Every function below is worked in full and
every test in this directory passes.  On `main` these are stubs raising
NotImplementedError, and making them pass is the exercise; read this file only
after you have had a go, or when you want to compare a method rather than an
answer.  Appendix C prints the answer and the tolerance, not the code.

The reference implementations of the same mathematics live in
`arith/scaling_budget.py`, which is what the chapter's printed numbers come
from.  Reading it before you have written your own is not forbidden, but it is
the whole exercise, so do it afterwards.

Having written them, these solutions then call that module rather than
carrying a second copy of the same five coefficients and the same objective.
A book with two implementations of one formula eventually prints two answers.
"""
import numpy as np

from arith import scaling_budget as sb


def fit_scaling_law(N, D, L, n_starts=200, seed=0):
    """E-10.10.  Fit (10.1) to a run table and report the whole ensemble.

    L(N, D) = L_inf + A N^-alpha + B D^-beta, five free parameters.

    Fit by minimising a Huber loss on the *logarithm* of the prediction, not on
    the prediction itself: the residuals in a run table are multiplicative, so
    a squared error in raw loss lets the two or three largest runs decide the
    answer.  §10.1 says why, and §10.5 is what happens when the fragility is
    not reported.

    Run the optimiser from `n_starts` random initialisations.  Return a dict:

        best        the five fitted values, keys "L_inf", "A", "alpha",
                    "B", "beta"
        objective   the best objective value reached
        all_fits    one dict per start, in the same key layout, each with its
                    own "objective"

    The point of returning every start and not only the winner is E-10.10's
    last question: which coefficient is least well determined?  You cannot
    answer that from one fit, and the curvature at the optimum will lie to you.

    The parametrisation is (log L_inf, log A, log B, alpha, beta) and the three
    terms of (10.1) are added by log-sum-exp, so the two coefficients stay
    positive by construction and the optimiser never has to be told so.  That
    is `sb._objective`, and it is called here rather than copied: the objective
    is what the chapter's fragility numbers were measured with, and a second
    transcription of it would be a second experiment.
    """
    N, D, L = np.asarray(N, float), np.asarray(D, float), np.asarray(L, float)
    objective = sb._objective(np.log(N), np.log(D), np.log(L))
    rng = np.random.default_rng(seed)

    all_fits = []
    for _ in range(n_starts):
        # log L_inf near the observed floor, the log coefficients anywhere over
        # twenty e-folds, the exponents anywhere in [0, 2].  Deliberately wide:
        # the count of starts that find the basin is one of the answers.
        x0 = np.array([rng.uniform(-1, 1), rng.uniform(0, 20), rng.uniform(0, 20),
                       rng.uniform(0, 2), rng.uniform(0, 2)])
        r = sb._minimise(objective, x0)
        e, a, b, alpha, beta = r.x
        all_fits.append({"L_inf": float(np.exp(e)), "A": float(np.exp(a)),
                         "alpha": float(alpha), "B": float(np.exp(b)),
                         "beta": float(beta), "objective": float(r.fun)})

    best = min(all_fits, key=lambda f: f["objective"])
    return {"best": {k: best[k] for k in ("L_inf", "A", "alpha", "B", "beta")},
            "objective": best["objective"], "all_fits": all_fits}


def isoflop_minimum(C, fit, n_grid=4001):
    """E-10.11.  The numerical minimum of one isoFLOP curve.

    At fixed compute C the budget constraint 6ND = C removes one variable, so
    the loss is a function of N alone.  Evaluate it on a log-spaced grid of N
    and return the minimising N.

    Return a float.  Refining the grid must move the answer toward (10.5), and
    that is what the test checks: agreement with the closed form to within 2%
    across five decades of C.

    Use a fixed grid wide enough to contain every optimum you will be asked
    for, N from 1e6 to 1e14, rather than one centred on where you expect the
    answer.  A grid placed using the closed form is not a check on the closed
    form, and an argmin sitting on an endpoint will agree with anything.
    """
    N = np.logspace(6, 14, n_grid)
    D = C / (6.0 * N)                     # the budget constraint, 6ND = C
    curve = sb.loss(N, D, fit)            # (10.1), the law itself
    return float(N[int(np.argmin(curve))])


def inference_aware_optimum(loss_target, D_inf, fit):
    """E-10.12.  Minimise lifetime compute 6ND + 2 N D_inf at fixed loss.

    Solve (10.8) numerically for N, then recover D from the loss constraint
    L(N, D) = loss_target.  Return a dict with keys "N", "D" and
    "tokens_per_param".

    Two properties the test pins, and both are the chapter's argument rather
    than an implementation detail:

      * at D_inf = 0 the answer collapses to D-10.2's compute-optimal frontier,
        so this is a strict generalisation and not a different model;
      * N is strictly decreasing and D/N strictly increasing in D_inf.  Serving
        more never argues for a larger model.

    Bracket carefully.  The loss constraint has no solution for N below
    (A / (loss_target - L_inf))^(1/alpha): that model cannot reach the target
    on any quantity of data, and the smallest feasible N is a pole, not a root.

    The solve is D-10.3 step 7: substitute D(N) from the loss constraint into
    the lifetime cost, differentiate, and root-find on the stationarity
    condition

        alpha A N^-alpha = beta B D^-beta (1 + D_inf / 3D),

    bracketed just above the pole and up to 1e13 parameters.  That is
    `sb.inference_aware_optimum`, which is what the chapter's serving numbers
    are printed from, so it is called rather than repeated.
    """
    return sb.inference_aware_optimum(loss_target, D_inf, fit)
