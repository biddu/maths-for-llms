"""Chapter 13 — Quantisation and Low-Rank Adaptation.

Generated into `notebooks/ch13_quant_lora.ipynb` by `build_all.py`.  The chapter
cites §1, §3, §4 and §5 by number, so sections may be added but never
renumbered.

The one thing this notebook must not do is agree with the folklore.  §5 checks
the scaling the chapter *corrected*: the adapter factor that cancels the rank is
alpha/sqrt(r), and alpha/r (which every tutorial prints) is wrong by the same
factor of r it claims to remove.
"""
from __future__ import annotations

CHAPTER = 13
SLUG = "quant_lora"
TITLE = "Quantisation and Low-Rank Adaptation"
BLURB = (
    "What a bit buys, what an outlier costs, and which part of the "
    "fine-tuning bill low-rank adaptation actually pays. Every format width, "
    "every memory total and every scaling exponent is recomputed from "
    "`arith/`, including the one the chapter corrected."
)

S1 = r'''
from arith.quant_formats import snr_db

SEED = 1301
rng = np.random.default_rng(SEED)


def affine(x, b_q, lo, hi):
    """Equation (13.1): scale, integer zero-point, clamp, dequantise."""
    s_q = (hi - lo) / (2 ** b_q - 1)
    z = np.round(-lo / s_q)
    q = np.clip(np.round(x / s_q) + z, 0, 2 ** b_q - 1)
    return s_q * (q - z), s_q


# Steps 1 to 3.  Inside the clamp the reconstruction error is bounded by half a
# step, and by nothing tighter: the bound is attained.
x = rng.uniform(-1.7, 2.3, 200_000)
for b_q in (2, 3, 4, 8):
    xhat, s_q = affine(x, b_q, x.min(), x.max())
    err = np.abs(x - xhat).max()
    assert err <= s_q / 2 + 1e-12, (b_q, err, s_q / 2)
    assert err > 0.49 * s_q, (b_q, err)          # attained, so not improvable
    print("b_q=%d: s_q = %.6f, worst |x - xhat| = %.6f = %.4f x s_q/2"
          % (b_q, s_q, err, err / (s_q / 2)))

# Steps 4 to 7.  The uniform-error model, then the decibel form.  6.02 is
# 10 log10 4 and nothing else, so it is asserted as that rather than typed.
assert abs(10 * np.log10(4.0) - 6.0206) < 1e-4
for b_q in (4, 8):
    xhat, s_q = affine(x, b_q, x.min(), x.max())
    e = xhat - x
    assert abs(e.var() / (s_q ** 2 / 12) - 1) < 0.02, (b_q, e.var())
assert abs(snr_db(8, 8.0) - snr_db(4, 8.0) - 4 * 6.02) < 1e-9
assert abs(snr_db(4, 8.0) - 16.81) < 0.01 and abs(snr_db(8, 8.0) - 40.89) < 0.01
print("(13.3) at a +/-4 s_x clip: %.2f dB at 4 bits, %.2f dB at 8, "
      "and exactly %.2f dB between them" % (snr_db(4, 8.0), snr_db(8, 8.0),
                                            snr_db(8, 8.0) - snr_db(4, 8.0)))

# Step 8 and the "what the model leaves out" note.  (13.3) counts rounding only,
# so it says a tighter clamp is always better.  It is not: clipping is the other
# error source, and the optimal clamp therefore DEPENDS ON THE BIT WIDTH.
g = rng.standard_normal(2_000_000)


def total_snr_db(b_q, clamp):
    xhat, _ = affine(g, b_q, -clamp, clamp)
    return 10 * np.log10(g.var() / ((xhat - g) ** 2).mean())


clamps = np.round(np.arange(1.5, 6.01, 0.1), 2)
best = {}
for b_q in (4, 8):
    curve = np.array([total_snr_db(b_q, c) for c in clamps])
    best[b_q] = (float(clamps[curve.argmax()]), float(curve.max()))
    print("b_q=%d: best clamp +/-%.1f s_x at %.2f dB; at +/-4 s_x, %.2f dB; "
          "(13.3) alone predicts %.2f dB"
          % (b_q, best[b_q][0], best[b_q][1], total_snr_db(b_q, 4.0),
             snr_db(b_q, 8.0)))

assert abs(best[4][0] - 2.5) < 1e-9 and abs(best[4][1] - 19.2) < 0.1
assert abs(total_snr_db(4, 4.0) - 16.3) < 0.1
assert abs(best[8][1] - 40.6) < 0.1 and best[8][0] >= 3.8
assert best[8][0] > best[4][0] + 1.0          # the optimum moves out with b_q
# and the rounding-only formula is optimistic at four bits, where clipping bites
assert snr_db(4, 8.0) > total_snr_db(4, 4.0)
print("the optimal clamp moves from +/-%.1f to +/-%.1f s_x between 4 and 8 "
      "bits: at low precision resolution is scarce and worth buying with a "
      "little clipping" % (best[4][0], best[8][0]))
'''

S2 = r'''
from arith.model_d import MODEL_D, total_params
from arith.quant_formats import (FORMATS, double_quantisation_saving,
                                 nf4_levels)

n = total_params(MODEL_D)

# (13.4), evaluated rather than quoted.  Nobody ships 4.000 bits per weight.
for f in FORMATS:
    predicted = f.b_q + (f.scale_bits + f.zero_bits) / f.g_q + f.second_bits
    assert abs(f.bits - predicted) < 1e-12, f.name
    assert abs(f.bytes_for(n) - n * f.bits / 8) < 1e-6

named = {f.name: f.bits for f in FORMATS if f.named}
assert abs(named["int4, fp16 scale per 128"] - 4.125) < 1e-12
assert abs(named["int4, fp16 scale and zero per 128"] - 4.25) < 1e-12
assert abs(named["MXFP4 (E8M0 scale per 32)"] - 4.25) < 1e-12
assert abs(named["NVFP4 (FP8 scale per 16)"] - 4.5) < 1e-12
lo, hi = min(named.values()), max(named.values())
spread = 100 * (hi / lo - 1)
assert abs(spread - 9.09) < 0.01, spread
gap_gb = n * (hi - lo) / 8 / 1e9
assert abs(gap_gb - 0.376) < 0.002, gap_gb
for name, bits in sorted(named.items(), key=lambda t: t[1]):
    print("%-40s %.4f bits  %6.3f GB on Model D" % (name, bits, n * bits / 8 / 1e9))
print("spread across shipped 4-bit formats: %.1f%% = %.3f GB, which decides "
      "whether an 8 B model fits a 24 GB card" % (spread, gap_gb))

# E-13.9: what the second level of scales is worth.  fp32 per block of 64 costs
# 0.5 bits per weight; fp8 per block plus fp32 per 256 blocks costs 0.127.
saving = double_quantisation_saving()
assert abs(saving - (32 / 64 - (8 / 64 + 32 / (256 * 64)))) < 1e-15
assert abs(saving - 0.373047) < 1e-6, saving
assert abs(4 + 8 / 64 + 32 / (256 * 64) - 4.127) < 1e-3
print("double quantisation saves %.5f bits/weight, %.0f MB on Model D"
      % (saving, n * saving / 8 / 1e6))

# NF4: sixteen normal quantiles, an exact zero, outermost level exactly one.
lv = nf4_levels()
assert lv.shape == (16,) and (np.diff(lv) > 0).all()
assert (lv == 0.0).sum() == 1                     # a format without one cannot
assert abs(lv.max() - 1.0) < 1e-12                # represent a pruned weight
assert abs(lv.min() + 1.0) < 1e-12
# equal probability mass per bin is what "normal float" means, so the levels
# crowd the centre: the inner gap is far smaller than the outer one.
gaps = np.diff(lv)
assert gaps[7] < 0.4 * gaps[0] and gaps[7] < 0.4 * gaps[-1]
print("NF4 levels: %s" % "  ".join("%+.3f" % v for v in lv))
print("innermost gap %.4f against outermost %.4f: equal mass, not equal width"
      % (gaps[7], gaps[0]))
'''

S3 = r'''
SEED = 1303
rng = np.random.default_rng(SEED)
d = 4096


def fwht(x):
    """The unnormalised Sylvester Walsh-Hadamard transform, log2(d) passes of
    butterflies and no multiplications at all."""
    x = np.asarray(x, dtype=float).copy()
    n = x.shape[0]
    h = 1
    while h < n:
        x = x.reshape(-1, 2, h)
        a, b = x[:, 0, :].copy(), x[:, 1, :].copy()
        x[:, 0, :] = a + b
        x[:, 1, :] = a - b
        x = x.reshape(n)
        h *= 2
    return x


def rotate(x, signs):
    """x Q with Q = diag(signs) H_d / sqrt(d), orthogonal for signs in {-1,+1}."""
    return fwht(x * signs) / np.sqrt(d)


def incoherence(v):
    return np.sqrt(len(v)) * np.abs(v).max() / np.linalg.norm(v)


ones = np.ones(d)

# Step 1: it is free, in exact arithmetic.  Step 2: the norm is preserved, so
# whatever the outlier carried is still there.
W = rng.standard_normal((d, 64)) / np.sqrt(d)
x = rng.standard_normal(d)
x[rng.choice(d, 12, replace=False)] *= 20.0
y = rotate(x, ones)
QtW = np.stack([rotate(W[:, j], ones) for j in range(W.shape[1])], axis=1)
assert abs(np.linalg.norm(y) - np.linalg.norm(x)) < 1e-9
assert np.abs(y @ QtW - x @ W).max() < 1e-9       # (xQ)(Q^T W) = xW

# Step 3: E[y_j^2] = ||x||^2 / d, exactly, because the transform is orthogonal
# and every coordinate of the image is a signed average of all of x.
assert abs((y ** 2).mean() * d / (x @ x) - 1.0) < 1e-12

# Steps 6 to 8.  The logarithm is not swept away: the gain is sqrt(d/2 ln d).
sq2lnd = np.sqrt(2 * np.log(d))
gain = np.sqrt(d / (2 * np.log(d)))
assert abs(sq2lnd - 4.0787) < 1e-3
assert abs(gain - 15.69) < 0.01
assert abs(np.log2(gain) - 3.97) < 0.01
assert gain < np.sqrt(d) / 4                      # not sqrt(d) = 64
before, after = incoherence(x), incoherence(y)
assert before > 15.0 and after < 2.0 * sq2lnd
print("spiked activation: incoherence %.1f -> %.2f, against sqrt(2 ln d) = "
      "%.3f" % (before, after, sq2lnd))
print("worst-case gain sqrt(d/2 ln d) = %.2f, that is %.2f bits, against the "
      "sqrt(d) = %.0f a reader might expect" % (gain, np.log2(gain), np.sqrt(d)))

# The assumption that does the work, and the failure the chapter names.  A FIXED
# Hadamard is derandomised, not random: a vector equal to one of its rows is
# mapped to a one-hot and the incoherence stays at its maximum sqrt(d).
row = fwht(np.eye(1, d, 7)[0]) / np.sqrt(d)
assert abs(np.linalg.norm(row) - 1.0) < 1e-12
fixed = incoherence(rotate(row, ones))
assert abs(fixed - np.sqrt(d)) < 1e-6, fixed
worst = max(incoherence(rotate(row, rng.choice([-1.0, 1.0], d)))
            for _ in range(200))
assert worst < 8.0, worst
assert fixed / worst > 8.0
print("on a vector aligned with a Hadamard row: fixed rotation leaves "
      "incoherence at %.1f (the maximum); with a random sign diagonal the "
      "worst of 200 draws is %.2f" % (fixed, worst))

# The detail that makes online rotation work: RMSNorm commutes with Q, because
# RMSNorm rescales by ||x||/sqrt(d) and an orthogonal map preserves the norm.
rms = lambda v: v / np.sqrt((v * v).mean())
signs = rng.choice([-1.0, 1.0], d)
assert np.abs(rotate(rms(x), signs) - rms(rotate(x, signs))).max() < 1e-10
# and the online cost, when fusion is impossible, is d log2 d additions
assert d * int(np.log2(d)) == 49152
print("RMSNorm(xQ) = RMSNorm(x)Q to %.1e, so the rotation passes through the "
      "normaliser; where it cannot be fused it costs %d additions per token "
      "per site" % (np.abs(rotate(rms(x), signs) - rms(rotate(x, signs))).max(),
                    d * int(np.log2(d))))
'''

S4 = r'''
from scipy.optimize import minimize

SEED = 1304
rng = np.random.default_rng(SEED)

# Steps 1 to 7 of D-13.3, checked against a constrained numerical minimisation
# rather than re-derived.  The objective is EXACTLY quadratic in the weights, so
# there is no approximation here to excuse a loose tolerance.
p = 8
A = rng.standard_normal((30, p))
H = 2 * A.T @ A
Hinv = np.linalg.inv(H)
for j in (0, 3, 7):
    delta = 0.37
    Delta = -delta / Hinv[j, j] * Hinv[:, j]
    assert abs(Delta[j] + delta) < 1e-12               # the constraint holds
    cost = 0.5 * Delta @ H @ Delta
    assert abs(cost - delta ** 2 / (2 * Hinv[j, j])) < 1e-10
    free = [i for i in range(p) if i != j]

    def obj(v, j=j, free=free, delta=delta):
        D = np.zeros(p)
        D[j] = -delta
        D[free] = v
        return 0.5 * D @ H @ D

    res = minimize(obj, np.zeros(p - 1), method="BFGS",
                   options={"gtol": 1e-12, "maxiter": 5000})
    assert abs(res.fun - cost) / cost < 1e-8, (j, res.fun, cost)
    print("j=%d: closed form %.8f, numerical minimum %.8f, and it is the "
          "diagonal of H INVERSE that appears" % (j, cost, res.fun))

# What the compensation needs, and it is not obvious.  (13.6) pushes the error
# along H^-1's j-th column, so a nearly diagonal H leaves nowhere to push.
N, K, M = 512, 512, 2048


def round_to_nearest(W, b_q=4, g_q=128):
    Q = np.empty_like(W)
    for s in range(0, W.shape[0], g_q):
        blk = W[s:s + g_q]
        lo, hi = blk.min(0), blk.max(0)
        sc = np.maximum((hi - lo) / (2 ** b_q - 1), 1e-12)
        z = np.round(-lo / sc)
        Q[s:s + g_q] = sc * (np.clip(np.round(blk / sc) + z, 0,
                                     2 ** b_q - 1) - z)
    return Q


def gptq(W, H, b_q=4, g_q=128, damp=0.01, act_order=False):
    """Step 8, left to right, with step 9's damping and the inverse Hessian
    downdated as coordinates are fixed."""
    n = W.shape[0]
    perm = np.arange(n)
    if act_order:
        perm = np.argsort(-np.diag(H))
        H, W = H[np.ix_(perm, perm)], W[perm]
    H = H + damp * np.mean(np.diag(H)) * np.eye(n)
    Hinv = np.linalg.inv(H)
    Wq, Q = W.copy(), np.zeros_like(W)
    for s in range(0, n, g_q):
        end = min(s + g_q, n)
        blk = Wq[s:end]
        lo, hi = blk.min(0), blk.max(0)
        sc = np.maximum((hi - lo) / (2 ** b_q - 1), 1e-12)
        z = np.round(-lo / sc)
        for j in range(s, end):
            w = Wq[j]
            Q[j] = sc * (np.clip(np.round(w / sc) + z, 0, 2 ** b_q - 1) - z)
            if j + 1 < n:
                # (13.6): SUBTRACTED.  The wrong sign is worse than doing
                # nothing by about as much as the right sign is better.
                Wq[j + 1:] -= np.outer(Hinv[j + 1:, j] / Hinv[j, j], w - Q[j])
                Hinv[j + 1:, j + 1:] -= (np.outer(Hinv[j + 1:, j],
                                                  Hinv[j, j + 1:]) / Hinv[j, j])
    return Q[np.argsort(perm)] if act_order else Q


def calibration(kind, seed=7):
    r = np.random.default_rng(seed)
    Z = r.standard_normal((M, N))
    if kind == "independent":
        X = Z
    else:                                   # a power-law covariance spectrum
        U, _ = np.linalg.qr(r.standard_normal((N, N)))
        X = (Z * (np.arange(1, N + 1) ** -0.5)) @ U.T
    X[:, r.choice(N, 15, replace=False)] *= 12.0
    return X


W = np.random.default_rng(41).standard_normal((N, K)) / np.sqrt(N)
gains = {}
for kind in ("independent", "correlated"):
    X = calibration(kind)
    H = 2 * X.T @ X
    e_rtn = np.linalg.norm(X @ W - X @ round_to_nearest(W))
    for order in (False, True):
        e_gptq = np.linalg.norm(X @ W - X @ gptq(W, H.copy(), act_order=order))
        gains[(kind, order)] = 1.0 - e_gptq / e_rtn
    Hi = np.linalg.inv(H + 0.01 * np.mean(np.diag(H)) * np.eye(N))
    diag = np.abs(np.diag(Hi)).sum()
    off = np.abs(Hi).sum() - diag
    print("%-12s off-diagonal mass of H^-1, per unit of diagonal: %.2f"
          % (kind, off / diag))

for key, g in gains.items():
    print("%-12s act_order=%-5s: %.1f%% better than round-to-nearest"
          % (key[0], key[1], 100 * g))

assert 0.03 < gains[("independent", False)] < 0.10
assert 0.10 < gains[("independent", True)] < 0.18
assert 0.36 < gains[("correlated", False)] < 0.46
assert 0.45 < gains[("correlated", True)] < 0.56
# the lesson, as an inequality rather than an anecdote
assert gains[("correlated", False)] > 4 * gains[("independent", False)]
assert gains[("correlated", True)] > gains[("correlated", False)]
print("calibrating off distribution does not merely make GPTQ suboptimal: it "
      "removes most of its reason to exist, %.0fx here"
      % (gains[("correlated", False)] / gains[("independent", False)]))

# and a compensated weight is still a quantised weight
Xc = calibration("correlated")
Q = gptq(W, 2 * Xc.T @ Xc)
for s in range(0, N, 128):
    for col in range(0, K, 97):
        assert len(np.unique(np.round(Q[s:s + 128, col], 10))) <= 16
'''

S5 = r'''
from arith.model_d import (MODEL_D, activation_bytes, finetune_memory,
                           lora_params, smallest_device, total_params)

SEED = 1355
chk = np.random.default_rng(SEED + 7)
d, alpha, eta, n_seeds = 4096, 16.0, 1e-3, 40
ranks = [4, 8, 16, 32, 64, 128, 256]

# Steps 3 to 5, first, on one small case: with A = 0 the gradient of B vanishes,
# A moves first, and one step of size eta changes the output by
# -eta gamma^2 ||u||^2 gbar.  This is checked against the arithmetic itself, not
# assumed, because everything below is a scaling argument about that expression.
dd, kk, r0, gamma0 = 32, 24, 6, 0.7
x0 = chk.standard_normal(dd)
B0 = chk.standard_normal((dd, r0)) / np.sqrt(dd)
A0 = np.zeros((r0, kk))
gbar = chk.standard_normal(kk)
u0 = x0 @ B0
gradA = gamma0 * np.outer(u0, gbar)
gradB = gamma0 * np.outer(x0, gbar @ A0.T)
assert np.abs(gradB).max() == 0.0                  # A = 0 kills the B gradient
dA = -eta * gradA
dy = gamma0 * (u0 @ dA)
assert np.abs(dy - (-eta * gamma0 ** 2 * (u0 @ u0) * gbar)).max() < 1e-14
print("first step: ||dy|| = eta gamma^2 ||u||^2 ||gbar||, checked to %.1e"
      % np.abs(dy - (-eta * gamma0 ** 2 * (u0 @ u0) * gbar)).max())

# Step 6: ||u||^2 is LINEAR in r, which is the whole mechanism.  Measured over
# 40 seeds at each rank, as the fraction ||u||^2 / ||x||^2 so that the draw of x
# divides out and only the rank dependence is left.
rng = np.random.default_rng(SEED)
S = {}
for r in ranks:
    tot = 0.0
    for _ in range(n_seeds):
        x = rng.standard_normal(d)
        B = rng.standard_normal((d, r)) / np.sqrt(d)
        u = x @ B
        tot += (u @ u) / (x @ x)
    S[r] = tot / n_seeds
    assert abs(S[r] * d / r - 1.0) < 0.25, (r, S[r] * d / r)
print("||u||^2 / ||x||^2 against the predicted r/d:")
for r in ranks:
    print("  r=%3d  measured %.6f  predicted %.6f  ratio %.4f"
          % (r, S[r], r / d, S[r] * d / r))

# Step 7.  Three scaling rules, over the same 64-fold range of r.  The step size
# is gamma(r)^2 ||u||^2, in units of eta ||gbar||.
rules = {"alpha/sqrt(r)": lambda r: alpha / np.sqrt(r),
         "alpha (constant)": lambda r: alpha,
         "alpha/r": lambda r: alpha / r}
step = {name: {r: g(r) ** 2 * S[r] for r in ranks} for name, g in rules.items()}
ratio = {name: step[name][256] / step[name][4] for name in rules}
for name in ("alpha/sqrt(r)", "alpha (constant)", "alpha/r"):
    print("gamma = %-16s step size changes by %9.4fx from r=4 to r=256"
          % (name, ratio[name]))

# The correction this chapter makes, as three assertions.  alpha/sqrt(r) is flat;
# alpha/r, the scaling every tutorial prints, is wrong by the same factor of r
# it claims to remove, in the other direction.
assert abs(ratio["alpha/sqrt(r)"] - 1.044) < 0.01, ratio["alpha/sqrt(r)"]
assert abs(ratio["alpha (constant)"] - 66.8) < 0.5, ratio["alpha (constant)"]
assert abs(ratio["alpha/r"] - 0.0163) < 0.0005, ratio["alpha/r"]
assert abs(ratio["alpha/sqrt(r)"] - 1.0) < 0.10          # flat to within 10%
# and the three are related exactly, because gamma^2 factors out of the mean
assert abs(ratio["alpha (constant)"] - 64 * ratio["alpha/sqrt(r)"]) < 1e-9
assert abs(ratio["alpha/r"] - ratio["alpha/sqrt(r)"] / 64) < 1e-12
assert ratio["alpha (constant)"] / ratio["alpha/r"] > 4000
print("only alpha/sqrt(r) is rank-invariant: a constant gamma takes steps "
      "%.0fx larger at r=256, and alpha/r takes them %.0fx smaller"
      % (ratio["alpha (constant)"], 1 / ratio["alpha/r"]))

# A-13.1.  What the adapter actually saves, and what it does not.  Grouped-query
# attention makes W_K and W_V d x (n_kv d_h), so an MHA-shaped count is high.
lp = lora_params(MODEL_D, r=16)
n = total_params(MODEL_D)
assert lp["per_layer"] == 16 * (2 * (MODEL_D.d + MODEL_D.h * MODEL_D.d_h)
                                + 2 * (MODEL_D.d + MODEL_D.n_kv * MODEL_D.d_h))
assert lp["total"] == lp["per_layer"] * MODEL_D.L == 13_631_488
assert abs(100 * lp["total"] / n - 0.170) < 0.001
assert abs(lp["mha_total"] / lp["total"] - 1.2308) < 1e-4
print("rank-16 adapters on all four projections: %.2f M parameters, %.3f%% of "
      "the model; an MHA-shaped count is %.0f%% high"
      % (lp["total"] / 1e6, 100 * lp["total"] / n,
         100 * (lp["mha_total"] / lp["total"] - 1)))

rows = finetune_memory(MODEL_D, r=16, b=8, s=4096)
act = activation_bytes(MODEL_D, b=8, s=4096)
for key in ("full", "lora", "qlora"):
    v = rows[key]
    assert v["act"] == act                # identical in all three regimes
    assert v["total"] == v["weights"] + v["state"] + v["act"]
    print("%-6s weights %6.2f GB  grad+opt %6.2f GB  activations %5.2f GB  "
          "total %6.1f GB  -> %s"
          % (key, v["weights"] / 1e9, v["state"] / 1e9, v["act"] / 1e9,
             v["total"] / 1e9, smallest_device(v["total"])))
assert abs(rows["full"]["total"] / 1e9 - 137.1) < 0.1
assert abs(rows["lora"]["total"] / 1e9 - 24.9) < 0.1
assert abs(rows["qlora"]["total"] / 1e9 - 12.9) < 0.1
assert smallest_device(rows["qlora"]["total"]) == "1 x 24 GB"
# the three terms move independently, which is the point of the table
assert rows["full"]["state"] > 100 * rows["lora"]["state"]      # LoRA's step
assert abs(rows["lora"]["weights"] / rows["qlora"]["weights"] - 3.88) < 0.02
print("full -> LoRA is optimiser state (%.0f GB -> %.2f GB); LoRA -> QLoRA is "
      "weights (%.1f GB -> %.1f GB); the %.1f GB of activations never moves"
      % (rows["full"]["state"] / 1e9, rows["lora"]["state"] / 1e9,
         rows["lora"]["weights"] / 1e9, rows["qlora"]["weights"] / 1e9,
         act / 1e9))
'''

SECTIONS = [
    ("1", "The error bound, and 6.02 dB per bit",
     "Affine quantisation rounds, so the reconstruction error is bounded by "
     "half a step and by nothing tighter. Modelling that error as uniform gives "
     "the decibel form, in which 6.02 is ten log ten of four and every other "
     "term is the loading factor. The last block is what the rounding-only "
     "model leaves out: clipping is the other error source, so the best clamp "
     "is not the widest one, and where it sits depends on the bit width.",
     S1),
    ("2", "What a four-bit model actually costs",
     "One scale per group is real memory, and the bits-per-weight formula is "
     "the whole of the format table. Evaluated at the settings people ship, no "
     "four-bit format is four bits, and the spread across them is nine per "
     "cent, which on an 8 B model decides whether a 24 GB card is enough. NF4 "
     "closes the section, because its levels are quantiles rather than a grid.",
     S2),
    ("3", "Rotation makes a spiked vector Gaussian",
     "The bits buy resolution against the range, so one coordinate at twenty "
     "times the median sets the range for all of them. An orthogonal map is "
     "free in exact arithmetic and redistributes the mass without destroying "
     "it, and the gain is the square root of d over two log d rather than the "
     "square root of d. The last block is the assumption that carries the "
     "argument: a fixed Hadamard is derandomised, not random, and on a vector "
     "aligned with one of its rows it accomplishes nothing at all.",
     S3),
    ("4", "Optimal brain surgeon, and GPTQ as its column sweep",
     "The layerwise objective is exactly quadratic in the quantised weights, so "
     "the optimal compensation for rounding one coordinate is a closed form and "
     "not an approximation. The cell checks it against a constrained numerical "
     "minimisation, then runs the column sweep on two calibration sets. The "
     "contrast is the lesson: the compensation travels along a column of the "
     "inverse Hessian, so on uncorrelated calibration data there is nowhere to "
     "push and the method has almost no advantage left.",
     S4),
    ("5", "What the scaling factor has to be",
     "With B Gaussian and A zero, one gradient step moves the output by a "
     "quantity proportional to gamma squared times the squared norm of xB, and "
     "that squared norm is linear in the rank. So the rank cancels exactly when "
     "gamma goes as one over the square root of r, and the conventional alpha "
     "over r is wrong by the same factor of r it claims to remove. The section "
     "closes with the memory table, where the three terms move independently "
     "and the activations do not move at all.",
     S5),
]
