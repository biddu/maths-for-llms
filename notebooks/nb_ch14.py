"""Chapter 14 — Generation and Decoding.

Generated into `notebooks/ch14_decoding.ipynb` by `build_all.py`.  The chapter's
three margin notes cite §1--§2, §3 and §4--§5, so this module has exactly three
sections and their numbers are ranges.  Renumber nothing.

Two places where the notebook must agree with the corrected chapter and not with
the folklore.  §1--2: BOTH nucleus rules widen monotonically in temperature, and
the widely repeated claim that top-p's does not is simply false.  §4--5: the
serving counter that divides accepted tokens by *drafted* tokens is not
measuring the acceptance rate at all.
"""
from __future__ import annotations

CHAPTER = 14
SLUG = "decoding"
TITLE = "Generation and Decoding"
BLURB = (
    "Temperature, truncation, beam search and speculation, checked against "
    "eight real logit positions spanning 0.001 to 6.09 nats and one real "
    "draft-target acceptance measurement, both committed under `figs/data/`."
)

# Repeated in each section on purpose: a cell that depends on an earlier cell's
# definitions is a cell that cannot be read on its own.
HELPERS = r'''
import os


def repo_file(*parts):
    """Works whether the notebook is run from notebooks/ or from the root."""
    for base in ("..", "."):
        p = os.path.join(base, *parts)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(os.path.join(*parts))


def temper(z, T):
    """softmax(z / T), computed stably."""
    a = z / T
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()


def entropy(p):
    q = p[p > 0]
    return float(-(q * np.log(q)).sum())
'''

S12 = HELPERS + r'''
SEED = 1412
rng = np.random.default_rng(SEED)

D = np.load(repo_file("figs", "data", "ch14_logits.npz"))
Z = D["logits"].astype(np.float64)
H_stored = D["entropy"].astype(np.float64)
V = Z.shape[1]
assert Z.shape[0] == 8 and V == 49152
assert np.abs([entropy(temper(z, 1.0)) for z in Z] - H_stored).max() < 1e-6
print("%d real positions, V = %d, entropy at T=1 from %.4f to %.4f nats"
      % (Z.shape[0], V, H_stored.min(), H_stored.max()))

# (14.2): temperature acts on the LOG-ODDS and on all of them by one factor.
# So it cannot reorder two tokens, cannot equalise them, and cannot remove one.
z = Z[4]
for T in (0.3, 0.7, 1.0, 2.5, 9.0):
    p = temper(z, T)
    lo = np.log(p[:64] / p[0])
    assert np.abs(lo - (z[:64] - z[0]) / T).max() < 1e-9
    assert (np.argsort(p) == np.argsort(z)).all()          # order preserved
    assert (p > 0).all()                                   # support preserved
gap = 10.0
assert abs(np.exp(gap / 0.5) - 4.85e8) < 1e7
print("temperature is a monotone map on the log-odds: a logit gap of %.0f at "
      "T = 0.5 is an odds ratio of %.2e, which is unlikely and not impossible"
      % (gap, np.exp(gap / 0.5)))

# D-14.1 steps 4 to 6.  dH/dT = Var_{p(T)}(z) / T^3, and the exponent is THREE.
Ts = np.geomspace(0.05, 20.0, 80)
for i, z in enumerate(Z):
    H = np.array([entropy(temper(z, T)) for T in Ts])
    assert (np.diff(H) > 0).all(), i                        # strictly monotone
    assert H[-1] < np.log(V) and H[0] >= 0.0
for T in (0.4, 1.0, 1.3, 4.0):
    for i, z in enumerate(Z):
        p = temper(z, T)
        var = float((p * (z - p @ z) ** 2).sum())
        h = 1e-4 * T
        num = (entropy(temper(z, T + h)) - entropy(temper(z, T - h))) / (2 * h)
        assert abs(var / T ** 3 - num) < 1e-4 * max(1.0, abs(num)), (T, i)
        if T != 1.0 and num > 1e-3:
            assert abs(var / T ** 2 - num) > 0.1 * num, (T, i)
print("dH/dT = Var_p(T)(z)/T^3 against a central difference at four "
      "temperatures and eight positions: T^2 agrees at T = 1 and nowhere else")

# (14.9): min-p is EXACTLY a window in logit space, of width T log(1/tau).
tau = 0.05
for i, z in enumerate(Z):
    for T in (0.7, 1.0, 1.5, 2.0, 5.0):
        p = temper(z, T)
        by_prob = p >= tau * p.max()
        by_logit = (z.max() - z) <= T * np.log(1 / tau)
        assert (by_prob == by_logit).all(), (i, T, (by_prob != by_logit).sum())
print("min-p's two forms agree exactly at every position and temperature: "
      "p_i >= tau p_max is z_max - z_i <= T log(1/tau), width %.3f at T = 1"
      % np.log(1 / tau))


def topp_size(z, T, thr):
    p = np.sort(temper(z, T))[::-1]
    return int(np.searchsorted(np.cumsum(p), thr) + 1)


def minp_size(z, T, tau):
    return int(((z.max() - z) <= T * np.log(1 / tau)).sum())


# (14.10): the mass of every top-k set FALLS as T rises, so the least k meeting
# a threshold rises.  BOTH rules widen monotonically, and the widely repeated
# claim that top-p's does not is false.  Checked on every real position.
for i, z in enumerate(Z):
    p = temper(z, 1.0)
    order = np.argsort(-p)
    for k in (1, 4, 40, 400):
        mass = [temper(z, T)[order[:k]].sum() for T in Ts]
        assert (np.diff(mass) < 1e-15).all(), (i, k)        # non-increasing
    a = np.array([topp_size(z, T, 0.9) for T in Ts])
    b = np.array([minp_size(z, T, 0.05) for T in Ts])
    assert (np.diff(a) >= 0).all(), ("top-p not monotone", i)
    assert (np.diff(b) >= 0).all(), ("min-p not monotone", i)
print("nucleus size is non-decreasing in T for top-p AND for min-p, on all "
      "%d positions over T in [%.2f, %.0f]" % (Z.shape[0], Ts[0], Ts[-1]))

# What separates them is the RATE, and the rate is a fact about real logits.
r_tp = np.array([topp_size(z, 1.5, 0.9) / topp_size(z, 0.7, 0.9) for z in Z])
r_mp = np.array([minp_size(z, 1.5, 0.05) / minp_size(z, 0.7, 0.05) for z in Z])
assert abs(np.median(r_tp) - 1.8) < 0.1 and abs(np.median(r_mp) - 1.8) < 0.1
assert abs(r_tp.max() - 261) < 2 and abs(r_mp.max() - 9) < 0.5
assert r_tp.max() > 25 * r_mp.max()
print("raising T from 0.7 to 1.5: top-p (0.9) grows by a median of %.1f and at "
      "worst %.0fx; min-p (0.05) by a median of %.1f and at worst %.1fx"
      % (np.median(r_tp), r_tp.max(), np.median(r_mp), r_mp.max()))

# and the received configuration equivalence fails exactly where it matters
pairs = [(topp_size(z, 1.0, 0.95), minp_size(z, 1.5, 0.05)) for z in Z]
assert pairs[0] == (1, 1) and pairs[-1] == (4689, 103)
assert abs(topp_size(Z[-1], 1.0, 0.95) - 40 - 4649) == 0     # top-k = 40 cuts
print("top_p=0.95 at T=1 against min_p=0.05 at T=1.5, by position: %s"
      % ", ".join("%d/%d" % pr for pr in pairs))
print("they agree to a token or two up to about 2 nats and read %d against %d "
      "at %.2f nats" % (pairs[-1][1], pairs[-1][0], H_stored[-1]))

# (14.7): the high-temperature rate, log V - H(p(T)) = sigma_z^2/2T^2 + O(T^-4).
sigma2 = Z.var(axis=1)
for T, tol in ((20.0, 0.03), (100.0, 0.01)):
    ratio = np.array([(np.log(V) - entropy(temper(z, T))) / (s / (2 * T ** 2))
                      for z, s in zip(Z, sigma2)])
    assert np.abs(ratio - 1).max() < tol, (T, np.abs(ratio - 1).max())
    print("T = %5.0f: measured gap over sigma_z^2/2T^2 in [%.4f, %.4f]"
          % (T, ratio.min(), ratio.max()))

# The order of the correction, measured rather than asserted.  The next term is
# odd in the logits, so on a symmetrised vector it vanishes and the residual
# falls by four for every doubling of T: that is the O(T^-4) of (14.7).
w = rng.standard_normal(512) * 1.9
zs = np.concatenate([w, -w])
assert abs(((zs - zs.mean()) ** 3).mean()) < 1e-12
dev = [abs(T ** 2 * (np.log(len(zs)) - entropy(temper(zs, T)))
           / (zs.var() / 2) - 1) for T in (10.0, 20.0, 40.0, 80.0)]
for a, b in zip(dev, dev[1:]):
    assert 3.5 < a / b < 4.5, (a, b)
print("on symmetrised logits the residual falls by %s for each doubling of T, "
      "so the correction is O(T^-4) and not O(T^-3)"
      % ", ".join("%.2fx" % (a / b) for a, b in zip(dev, dev[1:])))

# Two facts that separate temperature from truncation, since the interfaces do
# not.  No temperature reproduces a truncation: one equation fixes T and the
# next contradicts it.
p4 = np.array([0.5, 0.3, 0.15, 0.05])
trunc = np.array([0.625, 0.375, 0.0, 0.0])
z4 = np.log(p4)
T_match = (z4[0] - z4[1]) / (np.log(trunc[0]) - np.log(trunc[1]))
assert abs(T_match - 1.0) < 1e-12
assert abs(temper(z4, T_match)[2] - 0.15) < 1e-12 and trunc[2] == 0.0
print("matching the top two coordinates of a top_p=0.8 truncation forces "
      "T = %.4f exactly, and at that T the third coordinate is %.2f, not zero"
      % (T_match, temper(z4, T_match)[2]))
'''

S3 = HELPERS + r'''
from scipy import stats
from scipy.integrate import quad

SEED = 1413
rng = np.random.default_rng(SEED)


def scores(candidates, rho=0.6):
    """The three rules of D-14.2: (14.11), (14.13) and (14.14)."""
    out = {"unnormalised": [], "mean": [], "lp": []}
    for c in candidates:
        S, n = float(np.sum(c)), len(c)
        out["unnormalised"].append(S)
        out["mean"].append(S / n)
        out["lp"].append(S / ((5 + n) / 6) ** rho)
    return {k: np.array(v) for k, v in out.items()}


# Step 3.  Every term of (14.11) is at most zero, so every candidate dominates
# every one of its own extensions and EOS is structurally favoured.
logp = np.log(rng.uniform(0.35, 0.95, 40))
partial = np.cumsum(logp)
assert (np.diff(partial) < 0).all()
assert (logp <= 0).all()
print("the unnormalised score falls at every step, from %.3f at n=1 to %.3f at "
      "n=40: no extension can outscore its own prefix"
      % (partial[0], partial[-1]))

# Step 4.  At per-token entropy h the expected score is -hn, so a 40-token
# candidate must beat a 30-token one by about 10h nats to be ranked above it.
for h in (0.8, 1.5, 3.0):
    assert abs((-h * 40) - (-h * 30) + 10 * h) < 1e-12
print("at h = 1.5 nats a 40-token candidate must beat a 30-token one by %.0f "
      "nats, a criterion nobody chose" % (1.5 * 10))

# and under an exponential model of the per-token log-probabilities the
# probability the longer candidate wins does not depend on h at all, because h
# scales both sums by the same factor.
exact = quad(lambda t: stats.gamma.pdf(t, 40) * stats.gamma.sf(t, 30),
             0, 400, limit=400)[0]
assert 0.11 < exact < 0.12, exact
for h in (0.5, 1.5, 4.0):
    a = -h * rng.gamma(40, 1.0, 400_000)
    b = -h * rng.gamma(30, 1.0, 400_000)
    assert abs((a > b).mean() - exact) < 0.004, (h, (a > b).mean())
print("P(longer wins) = %.4f exactly, and %.4f, %.4f, %.4f by simulation at "
      "h = 0.5, 1.5, 4.0: independent of h"
      % (exact, *[(-h * rng.gamma(40, 1.0, 200_000)
                   > -h * rng.gamma(30, 1.0, 200_000)).mean()
                  for h in (0.5, 1.5, 4.0)]))

# Steps 5 and 6, on E-14.12's table.  The long candidate is uniformly better per
# token, so the disagreement is entirely about how length is charged for.
CANDIDATES = [[-0.34, -0.38],
              [-0.19, -0.21, -0.20, -0.22],
              [-0.15, -0.17, -0.16, -0.16, -0.16, -0.16]]
s = scores(CANDIDATES, rho=0.6)
for i, c in enumerate(CANDIDATES):
    n = len(c)
    assert abs(s["unnormalised"][i] - sum(c)) < 1e-12
    assert abs(s["mean"][i] - sum(c) / n) < 1e-12
    assert abs(s["lp"][i] - sum(c) / ((5 + n) / 6) ** 0.6) < 1e-12
assert int(np.argmax(s["unnormalised"])) == 0          # the shortest
assert int(np.argmax(s["mean"])) == 2                  # the longest
assert int(np.argmax(s["lp"])) == 1                    # the middle
print("three scorings of the same three candidates, %d, %d and %d tokens:"
      % tuple(len(c) for c in CANDIDATES))
for rule in ("unnormalised", "mean", "lp"):
    print("  %-13s %s  -> winner %d"
          % (rule, np.round(s[rule], 4), int(np.argmax(s[rule]))))

# Step 7: the correction's magnitude is TUNED, and the naming has teeth.  Both
# switches lie inside the band the literature recommends.
switches, prev = [], None
for rho in np.arange(0.0, 1.5005, 0.001):
    w = int(np.argmax(scores(CANDIDATES, rho=float(rho))["lp"]))
    if w != prev:
        switches.append((float(rho), w))
        prev = w
assert [w for _, w in switches] == [0, 1, 2]
assert abs(switches[1][0] - 0.52) < 0.01 and abs(switches[2][0] - 0.79) < 0.01
assert 0.6 <= switches[2][0] <= 1.0                    # inside the band
print("the lp(n) winner changes at rho = %.3f and rho = %.3f, and the "
      "recommended band [0.6, 1.0] straddles both: a tuned constant decides "
      "the answer" % (switches[1][0], switches[2][0]))

# Mean normalisation removes the linear term exactly, which is why it is a
# theorem and lp(n) is not.  It also over-corrects, and that is visible here.
lengths = np.arange(2, 60)
h = 1.5
assert np.abs(np.array([(-h * n) / n for n in lengths]) + h).max() < 1e-12
print("dividing by n leaves exactly -h at every length, so the length scale is "
      "gone; lp(n) leaves a residual that %s with n"
      % ("grows" if (-h * 60) / ((5 + 60) / 6) ** 0.6
         < (-h * 2) / ((5 + 2) / 6) ** 0.6 else "shrinks"))
'''

S45 = HELPERS + r'''
import json

from arith.decoding import (acceptance_counter, best_gamma, break_even_c,
                            bytes_per_emitted_token, plugin_bias, speedup,
                            tokens_per_round, verify_intensity_ratio)
from arith.model_d import MODEL_D, total_params

SEED = 1414
rng = np.random.default_rng(SEED)

# D-14.3 first, because the rest is only worth having if this holds.  The
# accept-reject rule emits EXACTLY the target distribution, for any draft.
Vs = 50
for trial in range(5):
    p = rng.dirichlet(np.full(Vs, 0.6))
    q = rng.dirichlet(np.full(Vs, 0.3))                 # a deliberately bad q
    tv = 0.5 * np.abs(p - q).sum()
    alpha = float(np.minimum(p, q).sum())
    assert abs(alpha - (1 - tv)) < 1e-12                # (14.15)
    resid = np.maximum(p - q, 0.0)
    assert abs(resid.sum() - tv) < 1e-12                # normaliser = P(reject)
    law = np.minimum(p, q) + resid                      # (14.17), pointwise
    assert np.abs(law - p).max() < 1e-14
print("acceptance probability = 1 - TV and the emission law is p pointwise, to "
      "%.1e, with no dependence whatever on the quality of q"
      % np.abs(law - p).max())

# and empirically, because the residual is the step people get wrong: replacing
# it by p gives min(p,q) + TV*p, which is a different distribution.
p = rng.dirichlet(np.full(Vs, 0.6))
q = rng.dirichlet(np.full(Vs, 0.3))
tv = 0.5 * np.abs(p - q).sum()
n = 200_000
x = rng.choice(Vs, n, p=q)
keep = rng.random(n) < np.minimum(1.0, p[x] / q[x])
resid = np.maximum(p - q, 0.0)
resid = resid / resid.sum()
emitted = np.where(keep, x, rng.choice(Vs, n, p=resid))
counts = np.bincount(emitted, minlength=Vs)
chi2 = float(((counts - n * p) ** 2 / (n * p)).sum())
from scipy.stats import chi2 as chi2_dist
pval = float(chi2_dist.sf(chi2, Vs - 1))
broken = np.where(keep, x, rng.choice(Vs, n, p=p))
cb = np.bincount(broken, minlength=Vs)
law_b = np.minimum(p, q) + tv * p
pval_b = float(chi2_dist.sf(((cb - n * p) ** 2 / (n * p)).sum(), Vs - 1))
assert pval > 0.01, pval
assert pval_b < 1e-6, pval_b
assert np.abs(cb / n - law_b).max() < 0.01
print("chi-squared against p: correct sampler p = %.3f, the one that resamples "
      "from p on rejection p = %.1e, and its law is min(p,q) + TV p" % (pval, pval_b))

# (14.18).  The tail sum, and the two checks of step 7.
for a in (0.0, 0.3, 0.65, 0.8, 0.9):
    for g in (0, 1, 2, 4, 8):
        t = tokens_per_round(a, g)
        assert abs(t - sum(a ** k for k in range(g + 1))) < 1e-12
        if g == 0 or a == 0.0:
            assert abs(t - 1.0) < 1e-12
assert abs(tokens_per_round(1.0, 4) - 5.0) < 1e-12
assert abs(tokens_per_round(0.8, 4) - 3.3616) < 1e-9
c = 1 / 8
assert abs(speedup(0.8, 4, c) - 2.2411) < 1e-4
assert abs(tokens_per_round(0.8, 4) / (1 + 4 * c) - speedup(0.8, 4, c)) < 1e-12
print("A-14.1: %.4f tokens per round over %.2f target-equivalents = %.4f, "
      "that is %.2fx" % (tokens_per_round(0.8, 4), 1 + 4 * c,
                         speedup(0.8, 4, c), speedup(0.8, 4, c)))
for a in (0.9, 0.8, 0.6):
    print("  alpha=%.1f: %.4f tokens, %.2fx" % (a, tokens_per_round(a, 4),
                                                speedup(a, 4, c)))
# alpha is the lever, not gamma: the numerator saturates at 1/(1-alpha)
assert speedup(0.8, 4, c) - speedup(0.6, 4, c) > speedup(0.8, 4, c) - speedup(0.8, 8, c)
for a in (0.5, 0.6, 0.7, 0.8, 0.9):
    g, s = best_gamma(a, c)
    # step 10: "draft about as many tokens as you expect to accept", within 20%
    assert abs(g - 1 / (1 - a)) / (1 / (1 - a)) <= 0.2 + 1e-9, (a, g)
    assert speedup(a, g, c) >= max(speedup(a, k, c) for k in range(1, 65))
gstar, sstar = best_gamma(0.8, c)
assert gstar == 5 and abs(1 / (1 - 0.8) - 5.0) < 1e-12
print("  optimal gamma = %d at %.2fx, against the rule of thumb 1/(1-alpha) = "
      "%.0f, which is within 20%% over alpha in [0.5, 0.9]"
      % (gstar, sstar, 1 / (1 - 0.8)))

# THE COUNTER TRAP.  A round drafts gamma tokens but TESTS only the positions up
# to and including the first rejection, so the two denominators differ.
for a in (0.6, 0.7, 0.8, 0.9):
    for g in (2, 4, 8):
        d = acceptance_counter(a, g)
        acc = sum(a ** k for k in range(1, g + 1))
        tested = sum(a ** k for k in range(0, g))
        assert abs(d["correct_counter"] - a) < 1e-12         # exactly alpha
        assert abs(d["correct_counter"] - acc / tested) < 1e-12
        assert abs(d["naive_counter"] - acc / g) < 1e-12
        assert abs(d["naive_counter"] - break_even_c(a, g)) < 1e-12
        assert d["naive_counter"] < a
d4 = acceptance_counter(0.8, 4)
d8 = acceptance_counter(0.8, 8)
assert abs(d4["naive_counter"] - 0.5904) < 1e-6
assert abs(100 * d4["understatement"] - 26.2) < 0.1
assert abs(100 * d8["understatement"] - 48.0) < 0.1
print("at alpha = 0.8, gamma = 4: accepted/tested = %.4f (exactly alpha), "
      "accepted/drafted = %.4f, understating by %.0f%%; at gamma = 8, by %.0f%%"
      % (d4["correct_counter"], d4["naive_counter"],
         100 * d4["understatement"], 100 * d8["understatement"]))
print("and %.4f is not an acceptance rate at all: it is (14.20)'s break-even "
      "draft cost ratio" % break_even_c(0.8, 4))

# The measurement, from the committed file rather than retyped.  One draft and
# one target, three workloads.
M = json.load(open(repo_file("figs", "data", "ch14_acceptance.json")))
alphas = {k: v["alpha"] for k, v in M.items()}
c_meas = [(v["tokens_plugin"] / v["speedup"] - 1) / 4 for v in M.values()]
assert max(c_meas) - min(c_meas) < 1e-6                 # one draft, one c
assert round(float(np.mean(c_meas)), 3) == 0.079
for name, v in M.items():
    assert abs(tokens_per_round(v["alpha"], 4) - v["tokens_plugin"]) < 1e-9
    assert abs(speedup(v["alpha"], 4, float(np.mean(c_meas))) - v["speedup"]) < 1e-6
    pb = plugin_bias(v["alpha"], v["sd"] ** 2, 4)
    assert abs(pb["jensen_correction"] - v["jensen"]) < 1e-9
    # BOTH corrections are visible: Jensen pushes the plug-in up, AM-GM pushes
    # it back down, and the truth sits between the two.
    assert v["tokens_plugin"] < v["tokens_true"] < pb["corrected"], name
    lo = 100 * (1 - v["tokens_plugin"] / v["tokens_true"])
    assert 1.0 <= lo <= 4.0, (name, lo)
    print("%-12s n=%3d  alpha %.3f (sd %.3f)  plug-in %.4f, true %.4f "
          "(%.1f%% low), speedup %.2fx"
          % (name, v["positions"], v["alpha"], v["sd"], v["tokens_plugin"],
             v["tokens_true"], lo, v["speedup"]))
spread = max(alphas.values()) - min(alphas.values())
swing = max(v["speedup"] for v in M.values()) - min(v["speedup"] for v in M.values())
assert abs(spread - 0.196) < 0.001 and abs(swing - 1.02) < 0.01
assert alphas["prose"] < alphas["code"] < alphas["boilerplate"]
print("a spread of %.3f in alpha across three workloads on ONE draft-target "
      "pair, worth %.2fx of speedup: quoting alpha from a paper is quoting "
      "someone else's traffic" % (spread, swing))

# The mechanism, in bytes.  At batch one the weights are re-read every round no
# matter how many tokens the round emits.
n = total_params(MODEL_D)
wb = 2 * n
assert abs(wb / 1e9 - 16.06) < 0.01
for a, g in ((0.8, 4), (0.9, 4), (0.8, 8)):
    b = bytes_per_emitted_token(wb, a, g)
    assert abs(b - wb / tokens_per_round(a, g)) < 1e-6
    print("alpha=%.1f gamma=%d: %.4f tokens per round, %.3f GB per emitted "
          "token against %.3f GB for plain decode" % (a, g, tokens_per_round(a, g),
                                                      b / 1e9, wb / 1e9))
assert bytes_per_emitted_token(wb, 0.8, 4) < wb / 3
assert verify_intensity_ratio(4) == 5.0
print("verifying gamma+1 positions in one pass multiplies the arithmetic by "
      "%.0f and leaves the weight traffic alone, which is why measured "
      "wall-clock beats this token-count model" % verify_intensity_ratio(4))
'''

SECTIONS = [
    ("1--2", "Temperature rescales log-odds; entropy is monotone in T",
     "Temperature divides the logits, so it acts on every log-odds by one "
     "factor and cannot reorder, equalise or remove a token. Differentiating "
     "the entropy gives a variance over T cubed, positive unless the logits are "
     "constant, and the exponent is three rather than the two a casual "
     "derivation produces. The cell then checks the two truncation rules on "
     "eight real positions: min-p is exactly a window in logit space, and both "
     "nucleus sizes widen monotonically in T, which is where the folklore is "
     "wrong. It closes with the high-temperature rate and the order of its "
     "correction.",
     S12),
    ("3", "Unnormalised beam search is biased towards short sequences",
     "The sequence score is a sum of non-positive terms, so every candidate "
     "outscores every one of its own extensions and end-of-sequence is "
     "structurally favoured. Dividing by the length removes the linear term "
     "exactly and is a theorem; the length penalty with its fitted five and six "
     "is a heuristic, and the cell shows the difference by finding the two "
     "values of rho at which the winner changes on a three-candidate table. "
     "Both switches lie inside the band the literature recommends, so the tuned "
     "constant decides the answer.",
     S3),
    ("4--5", "Expected tokens per verification round",
     "Speculative decoding is exact for any draft, so the draft affects speed "
     "and nothing else, and the cell checks that pointwise and again by a "
     "chi-squared test against the sampler people actually write by mistake. "
     "The expected token count is a tail sum, and the speedup divides it by one "
     "plus gamma times the draft cost. Then the trap: a round drafts gamma "
     "tokens but tests only up to the first rejection, so accepted over drafted "
     "is not the acceptance rate, it is the break-even cost ratio, and it "
     "understates alpha by a quarter at gamma equals four.",
     S45),
]
