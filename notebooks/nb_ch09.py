"""Chapter 9 — Optimisation.

Generated into `notebooks/ch09_optimisation.ipynb` by `build_all.py`.  The
chapter cites §1, §2, §3 and §4 by number, so sections may be added but never
renumbered.

The one section that needs care is §4.  The Newton-Schulz iteration of D-9.4 is
deliberately not convergent, so nothing here asserts that the iterate
approaches U V^T.  What is asserted is what the linear-minimisation oracle
actually needs: the singular vectors are preserved to machine precision, and
the singular values land in a band.
"""
from __future__ import annotations

CHAPTER = 9
SLUG = "optimisation"
TITLE = "Optimisation"
BLURB = (
    "An optimiser is a choice of geometry, and this notebook instantiates the "
    "template four times: the three linear-minimisation oracles, Adam's bias "
    "correction, the two spellings of weight decay, and the spectral-norm "
    "oracle that Muon computes with an odd polynomial."
)

# ---------------------------------------------------------------------------
# A brute-force check of a linear minimisation oracle.  Repeated in the section
# that needs it rather than defined once, because a cell that depends on an
# earlier cell's definitions cannot be read on its own.
S1 = r'''
from arith.model_d import MODEL_D

SEED = 9001
rng = np.random.default_rng(SEED)
n, eta = 64, 0.7
g = rng.standard_normal(n) * rng.gamma(2.0, size=n)   # a spread of magnitudes


def best_over_ball(g, project, trials=200000, seed=0):
    """The value of (9.2) by brute force: sample the ball, keep the smallest
    pairing.  A slow way to compute something the derivation gives in closed
    form, which is exactly why it is the right check."""
    r = np.random.default_rng(seed)
    U = project(r.standard_normal((trials, len(g))))
    return float((U @ g).min())


# ---- the l2 case.  Self-dual, so the value is -eta ||g||_2 and the minimiser
# is unique.
d2 = -eta * g / np.linalg.norm(g)
assert abs(np.linalg.norm(d2) - eta) < 1e-12
assert abs(g @ d2 + eta * np.linalg.norm(g)) < 1e-12
sampled = best_over_ball(g, lambda X: eta * X / np.linalg.norm(X, axis=1, keepdims=True),
                         seed=1)
assert sampled > g @ d2, (sampled, g @ d2)          # nothing feasible beats it
# uniqueness: every other feasible point of the same length is strictly worse
for _ in range(200):
    u = rng.standard_normal(n)
    u = eta * u / np.linalg.norm(u)
    assert g @ u > g @ d2 + 1e-12
print("l2  : value %.6f = -eta ||g||_2, best of 200k samples %.6f"
      % (g @ d2, sampled))

# ---- the l_inf case.  Dual is l1, and the minimiser is a VERTEX of the cube:
# every coordinate moves the same distance whatever its gradient.
dinf = -eta * np.sign(g)
assert np.abs(np.abs(dinf) - eta).max() < 1e-15     # a vertex, not a face
assert abs(g @ dinf + eta * np.abs(g).sum()) < 1e-12
sampled = best_over_ball(g, lambda X: eta * np.clip(X, -1, 1), seed=2)
assert sampled > g @ dinf
# the step size is uniform across coordinates although the gradients are not
assert np.abs(g).max() / np.abs(g).min() > 50
assert np.abs(dinf).max() == np.abs(dinf).min()
print("linf: value %.6f = -eta ||g||_1, uniform step %.3f although the "
      "gradients span %.0fx" % (g @ dinf, np.abs(dinf)[0],
                                np.abs(g).max() / np.abs(g).min()))

# ---- the P case.  A preconditioner is a norm, and its dual norm is the one
# built from P inverse.
p = rng.gamma(3.0, size=n) + 0.2                    # diag(P), positive definite
step = -np.linalg.solve(np.diag(p), g)              # proportional to -P^{-1} g
dP = eta * step / np.sqrt(step @ (p * step))        # rescaled to ||.||_P = eta
assert abs(np.sqrt(dP @ (p * dP)) - eta) < 1e-12
dual = np.sqrt(g @ (g / p))                         # ||g||_{P^{-1}}
assert abs(g @ dP + eta * dual) < 1e-12
sampled = best_over_ball(
    g, lambda X: eta * X / np.sqrt((X * (X * p)).sum(axis=1, keepdims=True)), seed=3)
assert sampled > g @ dP
# and setting P = I walks the same code back to the l2 case, which is the
# sentence about preconditioners being norms rather than heuristics
one = np.ones(n)
step_I = -np.linalg.solve(np.diag(one), g)
dI = eta * step_I / np.sqrt(step_I @ (one * step_I))
assert np.abs(dI - d2).max() < 1e-12
print("P   : value %.6f = -eta ||g||_{P^-1}, and P = I walks back to the l2 "
      "minimiser to %.1e" % (g @ dP, np.abs(dI - d2).max()))

# All three values are -eta times a dual norm, which is the statement of D-9.1.
for value, dualnorm in ((g @ d2, np.linalg.norm(g)),
                        (g @ dinf, np.abs(g).sum()),
                        (g @ dP, dual)):
    assert abs(value + eta * dualnorm) < 1e-12

# The failure mode, and it is a fact about geometry rather than about training.
# The cube's furthest point is a corner at eta sqrt(d) from the centre, so the
# linear model of (9.1) is trusted furthest exactly where it is weakest.
d = MODEL_D.d
assert np.linalg.norm(dinf) / eta == np.sqrt(n)
assert round(np.sqrt(d)) == 64 and d == 4096
print("at d = %d the l_inf corner is %.0fx further from the centre than the "
      "l2 sphere's surface" % (d, np.sqrt(d)))
'''

S2 = r'''
from arith.model_d import adam_burst_bound, beta2_half_life

SEED = 9002
rng = np.random.default_rng(SEED)
b1, b2 = 0.9, 0.999

# Step 1: the unrolled recursion, checked against the recursion itself.  m_0 = 0
# is the whole source of the bias, so it is spelled out rather than assumed.
T = 60
g = rng.standard_normal(T) * 0.3 + 1.4
m = 0.0
ms = []
for t in range(1, T + 1):
    m = b1 * m + (1 - b1) * g[t - 1]
    ms.append(m)
ms = np.array(ms)
unrolled = np.array([(1 - b1) * sum(b1 ** k * g[t - 1 - k] for k in range(t))
                     for t in range(1, T + 1)])
assert np.abs(ms - unrolled).max() < 1e-13
print("unrolled sum against the recursion: max abs %.2e" % np.abs(ms - unrolled).max())

# Steps 3 to 5.  With a constant gradient the expectation is exact, so the
# shrink factor can be read off with no sampling at all: it is the weight the
# empty prefix never received, not an approximation error.
const = 1.4
m = 0.0
for t in range(1, T + 1):
    m = b1 * m + (1 - b1) * const
    assert abs(m - (1 - b1 ** t) * const) < 1e-12, t
    assert abs(m / (1 - b1 ** t) - const) < 1e-12
print("constant gradient: m_t = (1 - b1^t) g exactly, for all %d steps" % T)

# and under a stationary but noisy gradient, by Monte Carlo over runs.
R, gbar = 40000, 1.4
G = rng.standard_normal((R, T)) * 0.9 + gbar
M = np.zeros(R)
V = np.zeros(R)
for t in range(1, T + 1):
    M = b1 * M + (1 - b1) * G[:, t - 1]
    V = b2 * V + (1 - b2) * G[:, t - 1] ** 2
    if t in (1, 3, 10, 60):
        raw, corrected = M.mean(), (M / (1 - b1 ** t)).mean()
        assert abs(raw - (1 - b1 ** t) * gbar) < 0.02, (t, raw)
        assert abs(corrected - gbar) < 0.02, (t, corrected)
        rv, cv = V.mean(), (V / (1 - b2 ** t)).mean()
        assert abs(cv - (gbar ** 2 + 0.81)) < 0.05, (t, cv)
        print("t = %2d: E[m] %.4f against (1 - b1^t) g = %.4f, corrected %.4f"
              % (t, raw, (1 - b1 ** t) * gbar, corrected))
# the raw estimate at t = 1 is shrunk by a full factor of ten, which is the
# size of the effect the correction exists to remove
assert abs((1 - b1 ** 1) - 0.1) < 1e-15

# Step 8: at t = 1 both corrections are exact and they cancel.  The first step
# of any run is a full-size sign step, whatever the gradient's magnitude.
for g1 in (1e-8, 1e-3, 1.0, 5.0, 1e4):
    for sgn in (+1.0, -1.0):
        m1 = (1 - b1) * (sgn * g1)
        v1 = (1 - b2) * g1 ** 2
        mhat, vhat = m1 / (1 - b1), v1 / (1 - b2)
        assert abs(mhat - sgn * g1) < 1e-18 * max(1.0, g1)
        assert abs(np.sqrt(vhat) - g1) < 1e-12 * max(1.0, g1)
        step = mhat / (np.sqrt(vhat) + 0.0)
        assert abs(step - sgn) < 1e-12, (g1, step)
print("first step: |m_hat / sqrt(v_hat)| = 1 exactly across ten decades of "
      "gradient magnitude")

# §9.5's two scales, from arith rather than from the page.  The averaging
# window of the second moment is what sets the warmup length, and the same
# quantity sets how long a spike takes to forget.
assert round(1 / (1 - b2)) == 1000
assert round(beta2_half_life(0.999)) == 693
assert round(beta2_half_life(0.95), 1) == 13.5
assert round(adam_burst_bound(b1, b2), 3) == 3.162
assert round(adam_burst_bound(b1, 0.95), 3) == 0.447
print("beta2 = %.3f: window %d steps, half-life %.0f steps, burst bound %.3f"
      % (b2, round(1 / (1 - b2)), beta2_half_life(b2), adam_burst_bound(b1, b2)))

# The failure mode.  Step 2 assumes a stationary mean.  A rapidly falling one,
# which is exactly what the opening of a run produces, makes it false: the
# average lags above the current mean and the correction, calibrated for a mean
# the window no longer has, inflates the lag rather than removing it.
decay = 4.0 * 0.85 ** np.arange(T)
m = 0.0
raw_err, cor_err = [], []
for t in range(1, T + 1):
    m = b1 * m + (1 - b1) * decay[t - 1]
    raw_err.append(abs(m - decay[t - 1]))
    cor_err.append(abs(m / (1 - b1 ** t) - decay[t - 1]))
raw_err, cor_err = np.array(raw_err), np.array(cor_err)
tail = slice(8, T)
assert (cor_err[tail] > raw_err[tail]).all()
assert (cor_err[tail] / np.maximum(raw_err[tail], 1e-12)).max() > 5.0
# and under the stationary mean of the Monte Carlo above the ordering is the
# other way round, which is why this is a failure of the assumption and not of
# the algebra
assert cor_err[0] < raw_err[0]
print("under a mean falling 15%% per step the uncorrected estimate is the "
      "better one for every step after the %dth, by up to %.0fx"
      % (tail.start, (cor_err[tail] / np.maximum(raw_err[tail], 1e-12)).max()))
'''

S3 = r'''
from arith.model_d import MODEL_D, optimiser_state, params_by_ndim

SEED = 9003
rng = np.random.default_rng(SEED)
n = 2048
eta, lam, eps = 3e-4, 0.1, 1e-8
b1, b2 = 0.9, 0.999

# A parameter vector whose coordinates receive gradients three decades apart,
# which is the situation the whole section is about.  A run in which every
# coordinate sees the same gradient scale cannot tell the two decays apart, and
# that coincidence is what step 6 explains.
w0 = 0.02 * rng.standard_normal(n)
scale = 10.0 ** rng.uniform(-4.0, -1.0, n)
assert scale.max() / scale.min() > 500

# ---- (9.13): decoupled decay is EXACTLY eta lambda w, whatever the moments
# hold.  Measured as the difference between two steps taken from one state, so
# the Adam part cancels rather than being argued away.
m_state = 0.7 * rng.gamma(2.0, size=n) * scale
v_state = rng.gamma(2.0, size=n) * scale ** 2
vhat = np.sqrt(v_state / (1 - b2 ** 5000))


def decoupled_step(vh, l):
    return w0 - eta * m_state / (vh + eps) - eta * l * w0


realised = decoupled_step(vhat, 0.0) - decoupled_step(vhat, lam)
rel = np.abs(realised - eta * lam * w0).max() / (eta * lam * np.abs(w0).max())
assert rel < 1e-9, rel
# and it does not depend on the preconditioner.  Scale the second moment by a
# thousand, which moves the Adam part by three decades, and the decay component
# is the same vector to the same tolerance.
other = decoupled_step(1000.0 * vhat, 0.0) - decoupled_step(1000.0 * vhat, lam)
assert np.abs(other - realised).max() / (eta * lam * np.abs(w0).max()) < 1e-9
print("decoupled: every coordinate is multiplied by (1 - eta lambda) = "
      "%.8f, whatever its gradient history (residual %.1e relative)"
      % (1 - eta * lam, rel))

# ---- (9.12): under an L2 penalty the same lambda arrives DIVIDED by the
# preconditioner.  Two facts make that precise, and they are checked apart.
#
# First, step 2.  The penalty gradient is inside m, and a constant term is
# reproduced by the EMA in full once the window has filled: it is not attenuated
# by (1 - beta_1), which is the reading that would make the effect ten times
# smaller than it is.
steps = 5000
gbar = scale * 1.0
m = np.zeros(n)
for t in range(1, steps + 1):
    m = b1 * m + (1 - b1) * (gbar + lam * w0)
mhat_l2 = m / (1 - b1 ** steps)
assert np.abs(mhat_l2 - (gbar + lam * w0)).max() / np.abs(gbar).max() < 1e-12
print("the penalty gradient enters m in full: |m_hat - (g + lambda w)| = %.1e"
      % np.abs(mhat_l2 - (gbar + lam * w0)).max())

# Second, step 3.  With the preconditioner held at its penalty-free value,
# which is the derivation's stated assumption, the realised decay is an
# identity rather than an approximation.
mhat_clean = gbar
update_l2 = eta * mhat_l2 / (vhat + eps)
update_clean = eta * mhat_clean / (vhat + eps)
predicted = eta * lam * w0 / (vhat + eps)
rel = np.abs((update_l2 - update_clean) - predicted).max() / np.abs(predicted).max()
assert rel < 1e-12, rel
print("L2: realised decay is eta lambda w / (sqrt(v_hat) + eps) to %.1e" % rel)

# ---- and the assumption itself, measured rather than asserted.  With
# zero-mean gradients the penalty's contamination of v is second order, so
# halving lambda quarters it; the ratio approaches four from below.
def contaminate(l, m_steps=8000, seed=41):
    r = np.random.default_rng(seed)
    v = np.zeros(n)
    for t in range(1, m_steps + 1):
        g = scale * r.standard_normal(n) + l * w0
        v = b2 * v + (1 - b2) * g * g
    return np.sqrt(v / (1 - b2 ** m_steps))


v_pure = contaminate(0.0)
ladder = (0.02, 0.01, 0.005, 0.0025, 0.00125)
sizes = [float(np.abs(contaminate(l) / v_pure - 1).mean()) for l in ladder]
ratios = [sizes[i] / sizes[i + 1] for i in range(len(sizes) - 1)]
assert all(2.5 < r_ < 4.0 for r_ in ratios), ratios
assert ratios == sorted(ratios), ratios               # approaching 4 from below
assert ratios[-1] > 3.5
print("contamination of v by the penalty, halving lambda four times: %s "
      "(ratios %s, so it is second order and the assumption holds)"
      % (["%.2e" % v for v in sizes], ["%.2f" % r_ for r_ in ratios]))

# ---- step 4, and it is the point.  Dividing by the second moment means the
# coordinates receiving the largest gradients are decayed LEAST, which is the
# opposite of the intent.
strength = 1.0 / (vhat + eps)
assert strength.max() / strength.min() > 100
assert np.corrcoef(np.log(scale), np.log(strength))[0, 1] < -0.98
lo, hi = np.percentile(vhat, [5, 95])
assert 1e-5 < lo and hi < 1e-1                # the O(1e-3) to O(1e-2) of §9.4
print("sqrt(v_hat) spans [%.2e, %.2e] across coordinates, so one lambda buys "
      "regularisation strengths %.0fx apart" % (lo, hi, strength.max() / strength.min()))

# ---- step 6: for SGD the two coincide EXACTLY, and that coincidence is the
# whole origin of the folklore that they are the same thing.
g = scale * rng.standard_normal(n)
sgd_l2 = w0 - eta * (g + lam * w0)
sgd_dec = (1 - eta * lam) * w0 - eta * g
assert np.abs(sgd_l2 - sgd_dec).max() < 1e-14 * np.abs(w0).max()
print("SGD: w - eta(g + lambda w) and (1 - eta lambda)w - eta g agree to %.1e, "
      "which is rounding and nothing else" % np.abs(sgd_l2 - sgd_dec).max())

# ---- the failure mode.  Transferring lambda between the two is not
# like-for-like: the ratio of the two realised decays is 1/(sqrt(v_hat) + eps),
# which here is two to three orders of magnitude.
transfer = (eta * lam * w0 / (vhat + eps)) / (eta * lam * w0)
assert 10 < np.percentile(transfer, 5) and np.percentile(transfer, 95) > 1e3
print("the same lambda means L2 decay %0.0fx to %0.0fx the decoupled one, "
      "across the 5th and 95th percentiles of coordinates"
      % (np.percentile(transfer, 5), np.percentile(transfer, 95)))

# ---- §9.8's ledger, from arith rather than retyped: what the optimiser
# costs, and how much of Model D is eligible for §9.7's alternative at all.
p = params_by_ndim(MODEL_D)
a = optimiser_state(MODEL_D, "adamw")
mu = optimiser_state(MODEL_D, "muon")
assert p["total"] == 8_030_261_248
assert p["two_d"] == 6_979_321_856 and p["other"] == 1_050_939_392
assert a["bytes_per_param"] == 16.0 and round(mu["bytes_per_param"], 2) == 12.52
assert round(100 * (1 - mu["state"] / a["state"])) == 29
assert round((a["state"] - mu["state"]) / 1e9, 1) == 27.9
print("AdamW holds %.2f GB against Muon's %.2f GB; %.2f B of %.2f B "
      "parameters are 2-D and therefore eligible"
      % (a["total"] / 1e9, mu["total"] / 1e9, p["two_d"] / 1e9, p["total"] / 1e9))
'''

S4 = r'''
import os
from scipy.optimize import brentq

from arith.model_d import newton_schulz_poly


def repo_file(*parts):
    """Works whether the notebook is run from notebooks/ or from the root."""
    for base in ("..", "."):
        q = os.path.join(base, *parts)
        if os.path.exists(q):
            return q
    raise FileNotFoundError(os.path.join(*parts))


D = np.load(repo_file("figs", "data", "fig94.npz"))
a, b, c = D["coeffs"]                       # the coefficients, not retyped
p = lambda s: a * s + b * s ** 3 + c * s ** 5
assert np.abs(p(np.linspace(0, 1.3, 41)) - newton_schulz_poly(np.linspace(0, 1.3, 41))).max() < 1e-12

SEED = 9004
rng = np.random.default_rng(SEED)
m_rows, n_cols, eta = 64, 200, 0.5
G = rng.standard_normal((m_rows, n_cols)) @ np.diag(
    10.0 ** rng.uniform(-2.0, 0.0, n_cols))       # a decaying spectrum on purpose
U, s, Vt = np.linalg.svd(G, full_matrices=False)

# ---- steps 1 to 4: the oracle itself.  Von Neumann's inequality caps the
# pairing at eta times the nuclear norm, and -eta U V^T attains it.
Delta = -eta * U @ Vt
assert abs(np.linalg.svd(Delta, compute_uv=False).max() - eta) < 1e-12
assert np.abs(np.linalg.svd(Delta, compute_uv=False) - eta).max() < 1e-12  # ALL of them
nuclear = s.sum()
assert abs(np.sum(G * Delta) + eta * nuclear) < 1e-10
# nothing in the spectral ball does better, checked by sampling rather than
# by trusting the inequality
worst = np.inf
for _ in range(4000):
    X = rng.standard_normal((m_rows, n_cols))
    X = eta * X / np.linalg.svd(X, compute_uv=False).max()
    worst = min(worst, float(np.sum(G * X)))
assert worst > np.sum(G * Delta)
print("spectral LMO: value %.6f = -eta ||G||_* ; best of 4000 samples %.6f"
      % (np.sum(G * Delta), worst))

# ---- step 5: the iteration, and what it does and does not do.
X = G / np.linalg.norm(G)
assert np.linalg.svd(X, compute_uv=False).max() <= 1.0 + 1e-12
for _ in range(5):
    A = X.T @ X
    X = a * X + b * (X @ A) + c * (X @ A @ A)

# The singular VECTORS are not approximately preserved, they are untouched:
# any odd polynomial in X acts on Sigma alone.
PU, PV = U @ U.T, Vt.T @ Vt
res_u = np.linalg.norm(X - PU @ X) / np.linalg.norm(X)
res_v = np.linalg.norm(X - X @ PV) / np.linalg.norm(X)
assert res_u < 1e-13 and res_v < 1e-13, (res_u, res_v)
# and the singular values are exactly the polynomial applied five times
sx = np.linalg.svd(X, compute_uv=False)
assert np.abs(np.sort(sx) - np.sort(newton_schulz_poly(s / np.linalg.norm(G), steps=5))).max() < 1e-12
print("after five steps the singular-vector residual is %.1e and %.1e; the "
      "singular values are p composed five times, to %.1e"
      % (res_u, res_v,
         np.abs(np.sort(sx) - np.sort(newton_schulz_poly(s / np.linalg.norm(G), steps=5))).max()))

# ---- step 7.  The polynomial is NOT converging to 1 and is not meant to.
# Its positive fixed points bracket where the iteration settles.
fp_lo = brentq(lambda t: p(t) - t, 0.5, 1.0, xtol=1e-13)
fp_hi = brentq(lambda t: p(t) - t, 1.1, 1.4, xtol=1e-13)
assert round(fp_lo, 3) == 0.868 and round(fp_hi, 3) == 1.264, (fp_lo, fp_hi)
assert p(0.0) == 0.0 and a > 3.0                # p'(0) = a, so small s climb fastest
rel = np.linalg.norm(X - U @ Vt) / np.linalg.norm(U @ Vt)
assert rel > 0.05, rel                          # it has NOT converged to U V^T
more = G / np.linalg.norm(G)
errs = {}
for k in range(1, 10):
    A = more.T @ more
    more = a * more + b * (more @ A) + c * (more @ A @ A)
    errs[k] = np.linalg.norm(more - U @ Vt) / np.linalg.norm(U @ Vt)
assert errs[7] >= errs[5] - 0.02 and errs[9] >= errs[5] - 0.02, errs
print("relative error against U V^T: %.3f at 5 steps, %.3f at 7, %.3f at 9. "
      "Fixed points %.4f and %.4f, so it oscillates between them."
      % (errs[5], errs[7], errs[9], fp_lo, fp_hi))

# ---- the band, on the committed spectrum of a real momentum matrix rather
# than on this cell's synthetic one.
before, after = D["s_before"], D["s_after"]
assert abs(np.linalg.norm(before) - 1.0) < 1e-9          # normalised by ||G||_F
assert round(before.sum() / before.max(), 1) == 17.7 and len(before) == 128
assert np.abs(np.sort(newton_schulz_poly(before, steps=5)) - np.sort(after)).max() < 1e-12
kept = before > 0.0016
assert kept.sum() == 122
band = newton_schulz_poly(before[kept], steps=5)
assert round(band.min(), 2) == 0.68 and round(band.max(), 2) == 1.20
assert 0.68 <= band.min() and band.max() <= 1.21
assert round(band.min(), 3) == 0.682
# and the lower edge is the polynomial's own oscillation, not the smallest
# input: starting anywhere in three decades lands in the same place
for s0 in (0.0016, 0.01, 0.05, 0.2, 0.5):
    assert 0.68 <= newton_schulz_poly(s0, steps=5) <= 1.21, s0
print("real momentum matrix: effective rank %.1f of %d; %d of %d singular "
      "values above 0.0016 land in [%.3f, %.3f], a factor of %.2f rather than "
      "the %.0f decades they started with"
      % (before.sum() / before.max(), len(before), kept.sum(), len(before),
         band.min(), band.max(), band.max() / band.min(),
         np.log10(before.max() / before.min())))

# ---- and why that is sufficient.  The oracle needs the DIRECTION, so what
# matters is how much of its value a step actually collects.  On the real
# spectrum above, and with every step rescaled to the same spectral radius so
# the comparison is like for like: the exact oracle collects the nuclear norm,
# five Newton-Schulz steps collect nearly three quarters of it, and the raw
# gradient direction under half.
p5 = newton_schulz_poly(before, steps=5)
value_oracle = before.sum()                                # = ||G||_*
value_ns = (before * p5).sum() / p5.max()
value_raw = (before * before).sum() / before.max()          # = ||G||_F^2/||G||_2
assert value_oracle > value_ns > value_raw > 0
assert round(value_ns / value_oracle, 2) == 0.73
assert round(value_raw / value_oracle, 2) == 0.45
assert value_ns / value_raw > 1.5
print("as a fraction of the oracle's value: five Newton-Schulz steps collect "
      "%.3f, the raw gradient direction only %.3f. Chasing s_i = 1 would buy "
      "the remaining %.0f%% and cost more iterations than it is worth."
      % (value_ns / value_oracle, value_raw / value_oracle,
         100 * (1 - value_ns / value_oracle)))
'''

SECTIONS = [
    ("1", "Steepest descent under a norm",
     "Minimising the linear model of equation (9.1) is meaningless until the "
     "step is confined, and the confinement is a modelling choice. The cell "
     "instantiates the oracle three times, checks each closed form against a "
     "brute-force search over the corresponding ball, and confirms that all "
     "three values are minus the radius times a dual norm. The last block is "
     "the failure mode: at Model D's width the cube's corner is 64 times "
     "further from the centre than the sphere's surface, so the linear model "
     "is trusted furthest exactly where it is weakest.",
     S1),
    ("2", "Adam's bias correction",
     "The shrink factor is not an approximation error, it is the weight the "
     "empty prefix never received, so with a constant gradient the unrolled "
     "recursion equals one minus beta to the t exactly. The cell checks that, "
     "then checks unbiasedness by Monte Carlo under a noisy but stationary "
     "mean, and then verifies step 8: at the first step both corrections are "
     "exact and cancel, so the very first update of any run is a full-size "
     "sign step whatever the gradient's magnitude. The last block is the "
     "failure mode, where a ramping mean makes the correction the worse "
     "estimator for several steps.",
     S2),
    ("3", "Decoupled weight decay",
     "An L2 penalty enters before the moment estimators, so it arrives at the "
     "parameter divided by the preconditioner, while decoupled decay is a "
     "clean multiplication by one minus eta lambda. The cell measures the "
     "realised per-coordinate decay under both spellings on a parameter vector "
     "whose coordinates receive gradients three decades apart, confirms that "
     "the two coincide exactly when the preconditioner is the identity (which "
     "is the whole origin of the folklore), and shows that under L2 the "
     "coordinates under the most fitting pressure are regularised least.",
     S3),
    ("4", "The spectral-norm LMO is UV^T, and an odd polynomial computes it",
     "Two claims, and only the first is an equality. The oracle's value is "
     "minus the radius times the nuclear norm and the minimiser is minus eta "
     "times U V transpose, checked here against a sampled search over the "
     "spectral ball. The Newton-Schulz iteration that computes it is "
     "deliberately not convergent, so nothing below asserts that it approaches "
     "U V transpose; what it does guarantee is that the singular vectors are "
     "untouched to machine precision and that the singular values are carried "
     "into a band. The band and the polynomial's two positive fixed points are "
     "measured on the committed spectrum of a real momentum matrix.",
     S4),
]
