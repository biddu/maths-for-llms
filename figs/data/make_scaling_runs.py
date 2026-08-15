"""Regenerate scaling_runs.csv, the run table Chapter 10 fits.

    python figs/data/make_scaling_runs.py [--check]

This table is synthetic and the book says so in §10.5's author note.  No
published sweep gives (N, D, loss) triples at enough points to fit five
parameters honestly, so the fragility experiment of §10.1 is run against a
table generated from the book's own frozen coefficients.  That is a weaker
claim than a replication and it is the right one: the experiment is about the
*fitting procedure*, and for that a table whose true answer is known is better
evidence than a real one whose answer is not.

The generating process, exactly:

  * nine model sizes, half a decade apart, 7e7 to 3e10 non-embedding
    parameters;
  * six token budgets per size at D/N in {5, 10, 20, 40, 80, 160}, which
    brackets both Chapter 10's compute-optimal 20 and Model D's shipped 2149;
  * runs below D = 2e9 tokens dropped, which is why the three smallest sizes
    carry fewer points.  A real sweep has the same shape: a small model trained
    on a handful of tokens is not a data point anyone bothers to produce;
  * loss = L(N, D) under REFIT_2024, multiplied by lognormal(0, 0.004) noise.
    Multiplicative, because run-to-run variation in a training loss is a
    percentage and not an absolute number, which is also why §10.1's fit works
    in log space.

Forty-eight rows, seed 10.  The committed CSV is reproduced byte for byte;
--check asserts it.
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from arith.model_d import REFIT_2024                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "scaling_runs.csv")

SIZES = [7e7, 1.5e8, 3e8, 7e8, 1.5e9, 3e9, 7e9, 1.5e10, 3e10]
RATIOS = [5, 10, 20, 40, 80, 160]
D_FLOOR = 2e9
SIGMA = 0.004
SEED = 10


def law(N: np.ndarray, D: np.ndarray, fit: dict | None = None) -> np.ndarray:
    f = fit or REFIT_2024
    return f["L_inf"] + f["A"] * N ** -f["alpha"] + f["B"] * D ** -f["beta"]


def table() -> list[tuple[float, float, float]]:
    grid = [(n, n * r) for n in SIZES for r in RATIOS if n * r >= D_FLOOR]
    N = np.array([g[0] for g in grid])
    D = np.array([g[1] for g in grid])
    noise = np.random.default_rng(SEED).lognormal(0.0, SIGMA, len(grid))
    L = law(N, D) * noise
    return list(zip(N, D, L))


def render(rows) -> str:
    out = ["N,D,loss"]
    out += [f"{n:.6e},{d:.6e},{l:.6f}" for n, d, l in rows]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="assert the committed file is reproduced exactly")
    args = ap.parse_args()
    text = render(table())
    if args.check:
        with open(CSV) as fh:
            on_disk = fh.read()
        assert text == on_disk, "generator no longer reproduces scaling_runs.csv"
        print(f"{CSV} reproduced exactly ({len(text.splitlines()) - 1} rows)")
        return
    with open(CSV, "w") as fh:
        fh.write(text)
    print(f"{CSV} written ({len(text.splitlines()) - 1} rows)")


if __name__ == "__main__":
    main()
