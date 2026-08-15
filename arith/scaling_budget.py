"""Scaling-law arithmetic for Chapter 10, and the book's frozen coefficient set.

Every scaling number the book prints comes from here.  No chapter quotes a
scaling-law coefficient from any other fit, and Chapter 8's loss of 2.03 nats
per token is this module evaluated at Model D's shape rather than a training
log.  If the coefficients move, both chapters re-run.

    python arith/scaling_budget.py            the Chapter 10 box
    python arith/scaling_budget.py --compare  the same table under both fits
    python arith/scaling_budget.py --fit      section 10.1's fragility numbers
"""
from __future__ import annotations
import argparse
import math

try:
    from arith.model_d import (MODEL_D, non_embedding, REFIT_2024, CHINCHILLA,
                               TRAINED_TOKENS)
except ImportError:                                  # run as a script from arith/
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from arith.model_d import (MODEL_D, non_embedding, REFIT_2024, CHINCHILLA,
                               TRAINED_TOKENS)

FITS = {"refit": REFIT_2024, "chinchilla": CHINCHILLA}


def loss(N: float, D: float, fit: dict | None = None) -> float:
    f = fit or REFIT_2024
    return f["L_inf"] + f["A"] * N ** -f["alpha"] + f["B"] * D ** -f["beta"]


def optimal_D(N: float, fit: dict | None = None) -> float:
    """D-10.2 step 4: at the optimum the two reducible terms are in fixed
    proportion, alpha A N^-alpha = beta B D^-beta.  Solve it for D."""
    f = fit or REFIT_2024
    lhs = f["alpha"] * f["A"] * N ** -f["alpha"]
    return (f["beta"] * f["B"] / lhs) ** (1.0 / f["beta"])


def optimal_N(C: float, fit: dict | None = None) -> float:
    """D-10.2 step 6: N* = (C / 6k)^(beta/(alpha+beta)) with k = (bB/aA)^(1/b)."""
    f = fit or REFIT_2024
    a, b, A, B = f["alpha"], f["beta"], f["A"], f["B"]
    k = (b * B / (a * A)) ** (1.0 / b)
    return (C / (6.0 * k)) ** (b / (a + b))


def tokens_per_param_exponent(fit: dict | None = None) -> dict[str, float]:
    """D-10.2 step 8.  The ratio drifts unless alpha = beta exactly."""
    f = fit or REFIT_2024
    a, b = f["alpha"], f["beta"]
    in_N = a / b - 1.0
    return {"in_C": (a - b) / (a + b), "in_N": in_N,
            "per_decade_of_N": 10.0 ** in_N - 1.0}


def flops_6nd(N: float, D: float) -> float:
    return 6.0 * N * D


def true_training_flops(N: float, D: float, s: int, c=MODEL_D) -> dict[str, float]:
    """D-10.1's three terms.  The attention score and value products carry no
    parameters, so 6ND does not see them, and the output projection is a real
    matmul that 6N misses when N is the non-embedding count."""
    per_token_params = 6.0 * N
    per_token_attn = 3.0 * 4.0 * s * c.d * c.L        # fwd 4sd per layer, x3 with bwd
    per_token_logit = 6.0 * c.d * c.V
    total = per_token_params + per_token_attn + per_token_logit
    return {"params": per_token_params * D, "attention": per_token_attn * D,
            "logits": per_token_logit * D, "total": total * D,
            "naive_6nd": per_token_params * D,
            "attn_ratio": per_token_attn / per_token_params,
            "logit_ratio": per_token_logit / per_token_params,
            "understatement": (total - per_token_params) / total}


def inference_aware_optimum(loss_target: float, D_inf: float,
                            fit: dict | None = None) -> dict[str, float]:
    """D-10.3.  Minimise 6ND + 2 N D_inf subject to L(N,D) = loss_target.

    The whole inference correction is the factor (1 + D_inf/3D) inflating the
    marginal value of data, so at D_inf = 0 this returns D-10.2's frontier.
    """
    from scipy.optimize import brentq
    f = fit or REFIT_2024
    a, b, A, B = f["alpha"], f["beta"], f["A"], f["B"]

    def D_at(N):                       # the loss constraint fixes D given N
        r = loss_target - f["L_inf"] - A * N ** -a
        if r <= 0:
            return float("inf")
        return (B / r) ** (1.0 / b)

    def stationarity(N):               # step 7 of D-10.3
        D = D_at(N)
        if not math.isfinite(D):
            return 1e9
        return a * A * N ** -a - b * B * D ** -b * (1 + D_inf / (3 * D))

    lo = (A / (loss_target - f["L_inf"])) ** (1 / a) * 1.0000001
    N = brentq(stationarity, lo, 1e13, xtol=1.0)
    D = D_at(N)
    return {"N": N, "D": D, "tokens_per_param": D / N,
            "train_flops": 6 * N * D, "serve_flops": 2 * N * D_inf,
            "lifetime_flops": 6 * N * D + 2 * N * D_inf}


def repeat_value(R: float, R_star: float = 15.4) -> dict[str, float]:
    """Muennighoff's decay: the R-th repeat is worth exp(-R/R*) of the first."""
    return {"effective_multiplier": 1 + R_star * (1 - math.exp(-R / R_star)),
            "marginal": math.exp(-R / R_star)}


def effective_params(N: float, bits: float, gamma: float = 1.1) -> float:
    """Kumar's precision-aware effective parameter count."""
    return N * (1 - math.exp(-bits / gamma))


def box(N: float | None = None, D_ship: float = TRAINED_TOKENS,
        fit: dict | None = None) -> dict[str, float]:
    """Chapter 10's arithmetic box, end to end."""
    from scipy.optimize import brentq
    f = fit or REFIT_2024
    N = non_embedding(MODEL_D) if N is None else N
    D_opt = optimal_D(N, f)
    C_opt, C_ship = 6 * N * D_opt, 6 * N * D_ship
    L_ship = loss(N, D_ship, f)
    N_c = brentq(lambda n: loss(n, optimal_D(n, f), f) - L_ship, 1e8, 1e13, xtol=1e3)
    D_c = optimal_D(N_c, f)
    extra = C_ship - 6 * N_c * D_c
    saved = 2 * (N_c - N)
    return {"N": N, "D_opt": D_opt, "C_opt": C_opt,
            "tokens_per_param_opt": D_opt / N,
            "D_ship": D_ship, "C_ship": C_ship,
            "tokens_per_param_ship": D_ship / N,
            "token_ratio": (D_ship / N) / (D_opt / N),
            "flop_ratio": C_ship / C_opt, "L_ship": L_ship,
            "N_c": N_c, "D_c": D_c, "C_c": 6 * N_c * D_c,
            "extra_train_flops": extra, "saved_per_token": saved,
            "break_even_tokens": extra / saved}


# ---------------------------------------------------------------------------
# Section 10.1: how fragile the fit is.  Two different questions, kept apart.
# ---------------------------------------------------------------------------

HUBER_DELTA = 1e-3        # Chinchilla's value, on log residuals


def _objective(lN, lD, lL):
    """Huber on the logarithm of the prediction, evaluated through a
    log-sum-exp so the three terms of (10.1) are added in log space and the
    parametrisation is (log L_inf, log A, log B, alpha, beta).  Fitting the raw
    loss instead lets the two or three largest runs decide the answer, because
    the residuals in a run table are multiplicative."""
    import numpy as np
    from scipy.special import logsumexp

    def obj(p):
        e, a, b, al, be = p
        pred = logsumexp(np.stack([np.full_like(lN, e), a - al * lN,
                                   b - be * lD]), axis=0)
        r = pred - lL
        x = np.abs(r)
        return float(np.where(x <= HUBER_DELTA, 0.5 * r * r,
                              HUBER_DELTA * (x - 0.5 * HUBER_DELTA)).sum())
    return obj


def _minimise(obj, x0):
    from scipy.optimize import minimize
    return minimize(obj, x0, method="L-BFGS-B",
                    bounds=[(-3, 2), (-5, 40), (-5, 40), (0, 2), (0, 2)],
                    options={"maxiter": 2000, "ftol": 1e-16, "gtol": 1e-12})


def run_table(path: str | None = None):
    """The committed run table of figs/data/scaling_runs.csv."""
    import csv
    import os
    import numpy as np
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "figs", "data", "scaling_runs.csv")
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return (np.array([float(r["N"]) for r in rows]),
            np.array([float(r["D"]) for r in rows]),
            np.array([float(r["loss"]) for r in rows]))


def fit_ensemble(n_starts: int = 200, seed: int = 0) -> dict:
    """Restart fragility: how often the optimiser finds the basin at all.

    This measures the solver, not the data.  The starts that succeed agree to
    the printed digit, so the spread among them says nothing.
    """
    import numpy as np
    N, D, L = run_table()
    obj = _objective(np.log(N), np.log(D), np.log(L))
    rng = np.random.default_rng(seed)
    xs, vs = [], []
    for _ in range(n_starts):
        x0 = np.array([rng.uniform(-1, 1), rng.uniform(0, 20), rng.uniform(0, 20),
                       rng.uniform(0, 2), rng.uniform(0, 2)])
        r = _minimise(obj, x0)
        xs.append(r.x)
        vs.append(float(r.fun))
    xs, vs = np.array(xs), np.array(vs)
    b = vs.min()
    p = xs[int(vs.argmin())]
    return {"best": {"L_inf": float(np.exp(p[0])), "A": float(np.exp(p[1])),
                     "alpha": float(p[3]), "B": float(np.exp(p[2])),
                     "beta": float(p[4])},
            "objective": b, "n_starts": n_starts,
            "n_converged": int((vs <= b * 1.01).sum()),
            "n_worse_than_2x": int((vs > 2 * b).sum()),
            "n_worse_than_10x": int((vs > 10 * b).sum())}


def sampling_spread(n_draws: int = 200, seed0: int = 1000) -> dict:
    """Sampling fragility: refit the same design under fresh run-to-run noise.

    This is the one that matters, and no amount of optimiser care removes it.
    Each draw starts from the known good point, so what is being measured is
    the data and not the search.
    """
    import numpy as np
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from figs.data.make_scaling_runs import SIZES, RATIOS, D_FLOOR, SIGMA, law

    grid = [(n, n * r) for n in SIZES for r in RATIOS if n * r >= D_FLOOR]
    N = np.array([g[0] for g in grid])
    D = np.array([g[1] for g in grid])
    x0 = np.array([math.log(1.829), math.log(503.9), math.log(2438.2),
                   0.3515, 0.3729])
    out = []
    for s in range(n_draws):
        L = law(N, D) * np.random.default_rng(seed0 + s).lognormal(0, SIGMA, len(grid))
        out.append(_minimise(_objective(np.log(N), np.log(D), np.log(L)), x0).x)
    X = np.array(out)
    cols = {"L_inf": np.exp(X[:, 0]), "A": np.exp(X[:, 1]), "alpha": X[:, 3],
            "B": np.exp(X[:, 2]), "beta": X[:, 4]}
    return {k: {"mean": float(v.mean()), "min": float(v.min()),
                "max": float(v.max()), "span": float(v.max() / v.min()),
                "spread_pct": float(100 * (v.max() - v.min()) / v.mean())}
            for k, v in cols.items()}


def fit_report() -> None:
    e = fit_ensemble()
    print(f"restart fragility, {e['n_starts']} random initialisations:")
    print(f"  reach the best objective   {e['n_converged']}")
    print(f"  land at more than 2x it    {e['n_worse_than_2x']}")
    print(f"  land at more than 10x it   {e['n_worse_than_10x']}")
    b, t = e["best"], REFIT_2024
    print("  best fit against the truth the table was generated from:")
    for k in ("L_inf", "A", "alpha", "B", "beta"):
        print(f"    {k:<6} {b[k]:10.4f}   truth {t[k]:10.4f}"
              f"   {100*(b[k]-t[k])/t[k]:+7.1f}%")
    print("\nsampling fragility, 200 draws of the same design:")
    for k, v in sampling_spread().items():
        print(f"  {k:<6} [{v['min']:9.4f}, {v['max']:9.4f}]"
              f"   x{v['span']:.2f}   mean {v['mean']:.4f}")


def report(compare: bool = False) -> None:
    b = box()
    print(f"Model D, N = {b['N']:.6e} non-embedding parameters")
    print(f"  compute-optimal D*   {b['D_opt']:.5e}  = {b['D_opt']/1e9:.0f} B tokens"
          f"   ({b['tokens_per_param_opt']:.2f} tokens/param)")
    print(f"  compute-optimal C    {b['C_opt']:.5e} FLOPs")
    print(f"  shipped D            {b['D_ship']:.2e}"
          f"   ({b['tokens_per_param_ship']:.1f} tokens/param)")
    print(f"  shipped C            {b['C_ship']:.5e} FLOPs")
    print(f"  token ratio          {b['token_ratio']:.2f}")
    print(f"  FLOP ratio           {b['flop_ratio']:.2f}"
          f"   (identical at fixed N: {abs(b['token_ratio']-b['flop_ratio']) < 1e-9})")
    print(f"  L(N, D_ship)         {b['L_ship']:.5f} nats/token")
    print(f"\n  the equal-loss compute-optimal model")
    print(f"    N_c {b['N_c']:.4e} ({b['N_c']/1e9:.1f} B)   D_c {b['D_c']:.4e}"
          f" ({b['D_c']/1e9:.0f} B)   C_c {b['C_c']:.4e}")
    print(f"    extra training FLOPs   {b['extra_train_flops']:.4e}")
    print(f"    saved per served token {b['saved_per_token']:.4e}")
    print(f"    break-even             {b['break_even_tokens']:.4e} tokens"
          f"  = {b['break_even_tokens']/3e10:.0f} days at 3e10/day")
    e = tokens_per_param_exponent()
    print(f"\n  D/N drifts as N^{e['in_N']:+.4f}"
          f"  ({100*e['per_decade_of_N']:+.1f}% per decade of N)")
    t = true_training_flops(b["N"], b["D_ship"], MODEL_D.trained_context)
    print(f"\n  6ND at s = {MODEL_D.trained_context}: attention is"
          f" {t['attn_ratio']:.3f}x and the logit head {t['logit_ratio']:.3f}x of 6N,")
    print(f"    so 6ND understates the true cost by {100*t['understatement']:.1f}%")
    if compare:
        print("\n  the same allocation under both fits:")
        for name, f in FITS.items():
            for n in (7e9, 7e10):
                d = optimal_D(n, f)
                print(f"    {name:<11} N = {n:.0e} : {d/n:7.1f} tokens/param")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--fit", action="store_true",
                    help="section 10.1's two fragility experiments")
    args = ap.parse_args()
    if args.fit:
        fit_report()
        return
    report(args.compare)


if __name__ == "__main__":
    main()
