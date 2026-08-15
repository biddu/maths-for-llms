"""Chapter 10 — Scaling Laws.

Generated into `notebooks/ch10_scaling.ipynb` by `build_all.py`.  The chapter
cites §1, §2 and §3 by number, so sections may be added but never renumbered.

Every coefficient here is the book's frozen 2024 refit, read from
`arith.model_d.REFIT_2024` rather than typed.  Chinchilla's published pair is
loaded alongside it and used only where the chapter contrasts the two, because
the two fits disagree about the SIGN of the tokens-per-parameter drift and that
disagreement is the point of §2.
"""
from __future__ import annotations

CHAPTER = 10
SLUG = "scaling"
TITLE = "Scaling Laws"
BLURB = (
    "Where 6ND comes from and by how much it is wrong, the closed-form "
    "compute-optimal allocation under the book's frozen refit, and what "
    "happens to that allocation once the model is also served."
)

S1 = r'''
from arith.model_d import MODEL_D, non_embedding, TRAINED_TOKENS
from arith.scaling_budget import flops_6nd, true_training_flops

SEED = 10001
rng = np.random.default_rng(SEED)

# ---- steps 1 to 5, counted rather than quoted.  One linear layer, tokens as
# rows, and the three matmuls Chapter 7 named.
d_in, d_out, s = 24, 17, 11
W = rng.standard_normal((d_in, d_out))
A = rng.standard_normal((s, d_in))
Ybar = rng.standard_normal((s, d_out))


def matmul_flops(a, b):
    """One multiply and one add per element of the accumulation, which is the
    convention the whole chapter uses and the reason the constant is 2."""
    assert a.shape[1] == b.shape[0]
    return 2 * a.shape[0] * a.shape[1] * b.shape[1]


forward = matmul_flops(A, W)                       # A W
back_a = matmul_flops(Ybar, W.T)                   # Ybar W^T
back_w = matmul_flops(A.T, Ybar)                   # A^T Ybar
params = W.size
assert forward == 2 * params * s
assert back_a == 2 * params * s and back_w == 2 * params * s
assert (forward + back_a + back_w) == 6 * params * s
print("one linear layer, %d tokens: forward %d, backward %d + %d, total "
      "%d = 6 x %d parameters x %d tokens"
      % (s, forward, back_a, back_w, forward + back_a + back_w, params, s))

# and 6ND is that statement summed over layers, so the per-token cost is 6N
# whatever the depth.  A three-layer stack, to make the "summed over layers"
# explicit rather than assumed.
Ws = [rng.standard_normal((12, 20)), rng.standard_normal((20, 20)),
      rng.standard_normal((20, 7))]
N_toy = sum(w.size for w in Ws)
total = 0
X = rng.standard_normal((s, 12))
for w in Ws:
    total += 3 * matmul_flops(X, w)                # forward once, backward twice
    X = X @ w
assert total == 6 * N_toy * s
assert flops_6nd(N_toy, s) == total
print("a three-layer stack of %d parameters over %d tokens: %d FLOPs = 6ND"
      % (N_toy, s, total))

# ---- step 6, the attention correction.  QK^T and AV carry NO parameters, so
# 6N cannot see them at all.  Counted directly at Model D's shape.
c = MODEL_D
N = non_embedding(c)
assert N == non_embedding(c) and 6.9e9 < N < 7.0e9   # non-embedding, not total
for s_ctx in (8192, 131072):
    per_token_scores = 2 * s_ctx * c.d            # q K^T, one row against s keys
    per_token_values = 2 * s_ctx * c.d            # the value mix
    fwd_per_layer = per_token_scores + per_token_values
    assert fwd_per_layer == 4 * s_ctx * c.d
    with_backward = 3 * fwd_per_layer * c.L
    t = true_training_flops(N, TRAINED_TOKENS, s_ctx)
    assert abs(t["attn_ratio"] - with_backward / (6 * N)) < 1e-12
print("attention FLOPs per token, forward and backward, over %d layers: "
      "12 L s d, and none of it is a parameter" % c.L)

# The two corrections at the trained context, and at the extended one.  These
# are the numbers the chapter's failure-mode paragraph prints.
t = true_training_flops(N, TRAINED_TOKENS, c.trained_context)
assert c.trained_context == 8192
assert round(t["attn_ratio"], 3) == 0.308
assert round(8192 / (6 * c.d), 3) == 0.333        # the idealisation it is not
assert round(t["logit_ratio"], 3) == 0.075
assert round(100 * t["understatement"], 1) == 27.7
assert t["naive_6nd"] == flops_6nd(N, TRAINED_TOKENS)
long = true_training_flops(N, TRAINED_TOKENS, c.extended_context)
assert round(long["attn_ratio"], 2) == 4.92
assert round(100 * long["understatement"]) == 83
print("s = %6d: attention %.1f%% and the logit head %.1f%% of 6N, so 6ND "
      "understates by %.1f%%" % (c.trained_context, 100 * t["attn_ratio"],
                                 100 * t["logit_ratio"],
                                 100 * t["understatement"]))
print("s = %6d: attention is %.2fx the parameter term, understatement %.0f%%"
      % (c.extended_context, long["attn_ratio"], 100 * long["understatement"]))

# ---- step 7, the embedding correction, separately: the input lookup is a
# gather and costs nothing, the output projection is a real matmul.
logit_per_token = 6 * c.d * c.V
assert abs(t["logit_ratio"] - logit_per_token / (6 * N)) < 1e-12
assert t["logits"] / TRAINED_TOKENS == logit_per_token
print("the unembedding costs 6 d V = %.3g FLOPs per token, which 6N misses "
      "because N is the non-embedding count" % logit_per_token)

# ---- step 8, the recompute correction.  Activation checkpointing replays the
# forward pass, so 6 becomes 8: a third more, and no published FLOP figure
# includes it.
assert 8 / 6 - 1 > 0.333 and round(100 * (8 / 6 - 1)) == 33
print("with activation checkpointing 6ND becomes 8ND, a %.0f%% increase"
      % (100 * (8 / 6 - 1)))

# ---- and the whole point, as one number.  The three corrections do not
# cancel, and a FLOP claim without a context length is not a claim.
assert t["total"] > t["naive_6nd"]
assert long["total"] / t["total"] > 4
print("the same N and D at two context lengths: %.3e against %.3e FLOPs, a "
      "factor of %.1f from s alone" % (t["total"], long["total"],
                                       long["total"] / t["total"]))
'''

S2 = r'''
from arith.model_d import REFIT_2024, CHINCHILLA, MODEL_D, non_embedding
from arith.scaling_budget import (loss, optimal_D, optimal_N, box,
                                  tokens_per_param_exponent)

# The frozen fit, read rather than retyped.  Chinchilla's published pair is
# loaded too, and used only where the chapter contrasts the two.
f = REFIT_2024
assert f["L_inf"] == 1.82 and f["A"] == 482.0 and f["alpha"] == 0.348
assert f["B"] == 2085.4 and f["beta"] == 0.366
print("the book's frozen 2024 refit: L_inf %.2f, A %.1f, alpha %.3f, B %.1f, "
      "beta %.3f" % (f["L_inf"], f["A"], f["alpha"], f["B"], f["beta"]))

# ---- step 4.  At the optimum the two REDUCIBLE terms sit in fixed
# proportion, and that single equation is the whole content of the allocation.
for N in (1e8, 7e9, 3e10, 1e12):
    D = optimal_D(N, f)
    lhs = f["alpha"] * f["A"] * N ** -f["alpha"]
    rhs = f["beta"] * f["B"] * D ** -f["beta"]
    assert abs(lhs / rhs - 1) < 1e-12, (N, lhs, rhs)
print("alpha A N^-alpha = beta B D^-beta holds to 1e-12 across four decades of N")

# ---- step 6, checked against the thing it claims to be: the minimiser of the
# isoFLOP curve.  Scan N at fixed C and confirm the closed form sits at the
# bottom, and that the curve really is U-shaped around it.
for C in (1e21, 1e23, 1e25):
    N_star = optimal_N(C, f)
    here = loss(N_star, C / (6 * N_star), f)
    grid = N_star * np.array([0.5, 0.8, 0.9, 0.95, 1.05, 1.1, 1.25, 2.0])
    around = np.array([loss(n, C / (6 * n), f) for n in grid])
    assert (around > here).all(), (C, around - here)
    fine = N_star * np.logspace(-0.6, 0.6, 401)
    curve = np.array([loss(n, C / (6 * n), f) for n in fine])
    assert abs(fine[curve.argmin()] / N_star - 1) < 5e-3
    print("C = %.0e: closed form N* = %.4e, grid minimum %.4e"
          % (C, N_star, fine[curve.argmin()]))

# and the two allocation exponents, recovered as slopes rather than asserted.
Cs = np.logspace(20, 26, 40)
Ns = np.array([optimal_N(C, f) for C in Cs])
Ds = Cs / (6 * Ns)
a, b = f["alpha"], f["beta"]
slope_N = np.polyfit(np.log(Cs), np.log(Ns), 1)[0]
slope_D = np.polyfit(np.log(Cs), np.log(Ds), 1)[0]
assert abs(slope_N - b / (a + b)) < 1e-9, (slope_N, b / (a + b))
assert abs(slope_D - a / (a + b)) < 1e-9
assert abs(slope_N + slope_D - 1.0) < 1e-9        # they must sum to one
print("N* goes as C^%.6f and D* as C^%.6f, against beta/(alpha+beta) = %.6f "
      "and alpha/(alpha+beta) = %.6f"
      % (slope_N, slope_D, b / (a + b), a / (a + b)))

# ---- step 1's quiet claim, worth its own check: L_inf drops out of every
# stationarity condition, so two fits disagreeing about the irreducible loss
# are not thereby disagreeing about where the optimum sits.
shifted = dict(f, L_inf=f["L_inf"] + 0.5)
for N in (1e9, 1e11):
    assert abs(optimal_D(N, shifted) / optimal_D(N, f) - 1) < 1e-15
assert abs(optimal_N(1e23, shifted) / optimal_N(1e23, f) - 1) < 1e-15
print("moving L_inf by 0.5 nats moves N* and D* by nothing at all")

# ---- step 8, and the correction that matters.  Under the frozen refit the
# tokens-per-parameter ratio FALLS with scale; under Chinchilla's published
# pair it RISES, and hard.  Both signs, because the chapter's failure-mode
# paragraph turns on the contrast.
e_refit = tokens_per_param_exponent(f)
e_chin = tokens_per_param_exponent(CHINCHILLA)
assert e_refit["in_N"] < 0 and e_chin["in_N"] > 0
assert round(e_refit["in_N"], 4) == -0.0492
assert round(100 * e_refit["per_decade_of_N"], 1) == -10.7
assert round(e_chin["in_N"], 3) == 0.214
assert round(100 * e_chin["per_decade_of_N"]) == 64
# and the same statement as two ratios a decade apart, under each fit
assert round(optimal_D(7e9, f) / 7e9, 1) == 20.6
assert round(optimal_D(7e10, f) / 7e10, 1) == 18.4
assert round(optimal_D(7e9, CHINCHILLA) / 7e9) == 67
assert round(optimal_D(7e10, CHINCHILLA) / 7e10) == 109
print("refit      : %.1f tokens/param at 7B, %.1f at 70B  (%.1f%% per decade)"
      % (optimal_D(7e9, f) / 7e9, optimal_D(7e10, f) / 7e10,
         100 * e_refit["per_decade_of_N"]))
print("chinchilla : %.0f tokens/param at 7B, %.0f at 70B  (%+.0f%% per decade)"
      % (optimal_D(7e9, CHINCHILLA) / 7e9, optimal_D(7e10, CHINCHILLA) / 7e10,
         100 * e_chin["per_decade_of_N"]))

# The exponent is zero exactly when alpha = beta, which is the punchline: twenty
# tokens per parameter is not an independent law, it is what a constant ratio
# looks like.  Force the two exponents equal and the drift vanishes.
equal = dict(f, beta=f["alpha"])
assert abs(tokens_per_param_exponent(equal)["in_N"]) < 1e-15
r = [optimal_D(n, equal) / n for n in (1e8, 1e10, 1e12)]
assert max(r) / min(r) - 1 < 1e-12
print("with alpha = beta the ratio is %.2f tokens per parameter at every "
      "scale, and the VALUE comes from A and B rather than from the exponents"
      % r[0])

# ---- §10.8's box, end to end, from arith.  Model D ships far off the
# compute-optimal frontier and the chapter's whole §10.4 is about why that is
# rational.
bx = box()
assert bx["N"] == non_embedding(MODEL_D)
assert round(bx["tokens_per_param_opt"], 2) == 20.60
assert round(bx["tokens_per_param_ship"], 1) == 2149.1
assert round(bx["token_ratio"], 2) == 104.34
assert abs(bx["token_ratio"] - bx["flop_ratio"]) < 1e-9    # one number, not two
assert round(bx["L_ship"], 5) == 2.03227                   # Chapter 8's 2.03
assert round(bx["break_even_tokens"] / 3e10) == 378
print("Model D: %.2f tokens/param optimal against %.1f shipped, a factor of "
      "%.2f, at a predicted loss of %.5f nats/token"
      % (bx["tokens_per_param_opt"], bx["tokens_per_param_ship"],
         bx["token_ratio"], bx["L_ship"]))
'''

S3 = r'''
from arith.model_d import REFIT_2024
from arith.scaling_budget import inference_aware_optimum, optimal_D, loss

f = REFIT_2024
TARGET = 2.20                       # one fixed loss, held across the whole cell

# ---- step 7, and the check the chapter says is worth making.  At
# D_inf = 0 the bracket is one and this is D-10.2's frontier exactly, so the
# inference-aware optimum is not a different theory.
base = inference_aware_optimum(TARGET, 0.0, f)
assert abs(base["D"] / optimal_D(base["N"], f) - 1) < 1e-4
assert abs(loss(base["N"], base["D"], f) - TARGET) < 1e-6
assert base["serve_flops"] == 0.0
print("D_inf = 0 reproduces the training-only frontier: D = %.4e against "
      "optimal_D(N*) = %.4e" % (base["D"], optimal_D(base["N"], f)))

# ---- the monotonicity that is the whole content of §10.6.  As the served
# token count rises, N* falls strictly and D*/N* rises strictly, and the
# constraint is satisfied throughout, so every row is the same model quality.
ladder = [0.0, 1e11, 1e12, 1e13, 1e14, 1e15]
rows = [inference_aware_optimum(TARGET, D_inf, f) for D_inf in ladder]
for r in rows:
    assert abs(loss(r["N"], r["D"], f) - TARGET) < 1e-6
Ns = [r["N"] for r in rows]
ratios = [r["tokens_per_param"] for r in rows]
assert all(Ns[i] > Ns[i + 1] for i in range(len(Ns) - 1)), Ns
assert all(ratios[i] < ratios[i + 1] for i in range(len(ratios) - 1)), ratios
assert ratios[-1] / ratios[0] > 1000
print("%-10s %-12s %-12s %-10s" % ("D_inf", "N*", "D*", "D*/N*"))
for D_inf, r in zip(ladder, rows):
    print("%-10.0e %-12.4e %-12.4e %-10.1f"
          % (D_inf, r["N"], r["D"], r["tokens_per_param"]))

# ---- and it really is an optimum of lifetime FLOPs, not just a stationary
# point of an equation.  Perturb N along the fixed-loss constraint and the
# lifetime cost goes up in both directions.
def D_at(N):
    resid = TARGET - f["L_inf"] - f["A"] * N ** -f["alpha"]
    assert resid > 0
    return (f["B"] / resid) ** (1.0 / f["beta"])


for D_inf in (1e12, 1e14):
    r = inference_aware_optimum(TARGET, D_inf, f)
    here = 6 * r["N"] * r["D"] + 2 * r["N"] * D_inf
    assert abs(here - r["lifetime_flops"]) < 1e-6 * here
    for k in (0.7, 0.9, 0.98, 1.02, 1.1, 1.4):
        n = r["N"] * k
        cost = 6 * n * D_at(n) + 2 * n * D_inf
        assert cost > here, (D_inf, k, cost, here)
    # and the training-only allocation is strictly worse once serving is paid for
    naive = base["N"]
    assert 6 * naive * D_at(naive) + 2 * naive * D_inf > here
    print("D_inf = %.0e: lifetime %.4e FLOPs at N* = %.4e, against %.4e for "
          "the training-only allocation"
          % (D_inf, here, r["N"],
             6 * naive * D_at(naive) + 2 * naive * D_inf))

# ---- reading the correction.  The entire change is the factor
# (1 + D_inf/3D) inflating the marginal value of data, and it is what turns
# twenty tokens per parameter into a few thousand.
# "Ten times your training tokens" is a statement about the training tokens
# AT the optimum, so it is a fixed point rather than a substitution: solve for
# the D_inf at which D_inf = 10 D*.
from scipy.optimize import brentq

D_inf_star = brentq(
    lambda x: inference_aware_optimum(TARGET, 10 * x, f)["D"] - x, 1e11, 1e13)
r10 = inference_aware_optimum(TARGET, 10 * D_inf_star, f)
assert abs(r10["D"] / D_inf_star - 1) < 1e-6
factor = 1 + (10 * D_inf_star) / (3 * r10["D"])
lhs = f["alpha"] * f["A"] * r10["N"] ** -f["alpha"]
rhs = f["beta"] * f["B"] * r10["D"] ** -f["beta"] * factor
assert abs(lhs / rhs - 1) < 1e-6                  # equation (10.14) itself
assert abs(factor - (1 + 10 / 3)) < 1e-5          # exactly 4.333 at D_inf = 10 D
assert round(factor, 2) == 4.33
assert round(factor ** (1 / f["alpha"])) == 68
assert 1000 < r10["tokens_per_param"] < 1500
print("serving ten times the training tokens makes the bracket %.3f, which "
      "raises the ratio by %.0fx, from %.1f to %.0f tokens per parameter"
      % (factor, factor ** (1 / f["alpha"]), base["tokens_per_param"],
         r10["tokens_per_param"]))

# ---- and the failure mode, which is the assumption that bites: D_inf must be
# known in advance.  The chapter's 10^(1/alpha) is the sensitivity of (10.14)
# read at fixed D, and it is enormous.
D_fixed = rows[4]["D"]


def N_from_balance(D_inf, D=D_fixed):
    """Solve (10.14) for N with D held where it is."""
    rhs = f["beta"] * f["B"] * D ** -f["beta"] * (1 + D_inf / (3 * D))
    return (f["alpha"] * f["A"] / rhs) ** (1 / f["alpha"])


for D_inf in (1e17, 1e18):                        # deep in the D_inf >> 3D limit
    moved = N_from_balance(D_inf) / N_from_balance(10 * D_inf)
    assert abs(moved / 10 ** (1 / f["alpha"]) - 1) < 1e-3, moved
assert round(10 ** (1 / f["alpha"])) == 747
print("at fixed D, a factor of ten in D_inf moves D/N by 10^(1/alpha) = %.0fx"
      % 10 ** (1 / f["alpha"]))

# Along the constrained frontier, where D moves too, the realised sensitivity
# is gentler and remarkably steady: the same factor of ten in D_inf costs about
# a factor of six in the ratio, decade after decade.
walked = [inference_aware_optimum(TARGET, x, f)["tokens_per_param"]
          for x in (1e13, 1e14, 1e15)]
steps = [walked[1] / walked[0], walked[2] / walked[1]]
assert all(5.0 < x < 7.0 for x in steps), steps
assert abs(steps[0] / steps[1] - 1) < 0.05        # the same factor each decade
print("along the frontier, where D moves with it, the same factor of ten "
      "costs %.2fx and then %.2fx" % (steps[0], steps[1]))
'''

SECTIONS = [
    ("1", "Why training costs 6ND, and when it does not",
     "Six FLOPs per parameter per token is two for the forward pass and four "
     "for the backward, and the cell counts all three matmuls on a real layer "
     "rather than quoting the constant. The three corrections then follow: "
     "attention carries no parameters so 6N cannot see it, the output "
     "projection is a genuine matmul that a non-embedding N misses, and "
     "activation checkpointing replays the forward pass. At Model D's trained "
     "context 6ND understates the true cost by a quarter, and at the extended "
     "context by more than four fifths.",
     S1),
    ("2", "Compute-optimal N* and D*",
     "One equation carries the whole allocation: at the optimum the two "
     "reducible loss terms sit in fixed proportion. The cell checks that "
     "balance, confirms that the closed form really is the minimiser of the "
     "isoFLOP curve by scanning it, and recovers both allocation exponents as "
     "slopes. The coefficients are the book's frozen 2024 refit and not "
     "Chinchilla's published pair, and the difference is not cosmetic: under "
     "the refit the tokens-per-parameter ratio falls with scale, under the "
     "published pair it rises by two thirds per decade, and the cell asserts "
     "both signs.",
     S2),
    ("3", "The inference-aware optimum",
     "Serving costs two FLOPs per parameter per generated token, so once "
     "lifetime compute is the objective the optimum moves. The whole "
     "correction is one bracket inflating the marginal value of data, and at "
     "no serving at all it is one and the frontier is the previous section's. "
     "The cell walks a ladder of served token counts and asserts what the "
     "chapter claims: the optimal parameter count falls strictly, the "
     "tokens-per-parameter ratio rises strictly, and the loss is held fixed "
     "throughout so every row is the same model quality bought a different way.",
     S3),
]
