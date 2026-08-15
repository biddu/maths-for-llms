"""Chapter 16 — What the Model Represents.

Generated into `notebooks/ch16_representation.ipynb` by `build_all.py`.  No
margin note in the chapter fixes a section number here, so the four sections
follow the four derivations in order and should stay that way.

Two places where the notebook must agree with the corrected chapter rather than
with the draft it replaced.  §2: the union bound over m^2 nearly independent
events is loose by a SINGLE-DIGIT factor, not by two or three orders of
magnitude.  §3: the interference standard deviation is exactly sqrt((k-1)/d), so
at d = 4096 and k = 200 the separation is still about four and a half standard
deviations and the reads are trivially separable.
"""
from __future__ import annotations

CHAPTER = 16
SLUG = "representation"
TITLE = "What the Model Represents"
BLURB = (
    "Whether adding a vector to the residual stream is an edit or damage, how "
    "many directions a stream of width d holds, what interference between them "
    "costs, and what the dictionary that decodes them costs. A bound, a price "
    "and a limit, all three measured rather than quoted."
)

S1 = r'''
SEED = 1601
rng = np.random.default_rng(SEED)
d, L = 64, 12

# Step 1.  A pre-norm block is a + F(N(a)), so unrolling is an IDENTITY and not
# an approximation: the stream is the embedding plus a sum of writes.
def rmsnorm(a):
    return a / np.sqrt((a * a).mean())


Ws = [rng.standard_normal((d, d)) / np.sqrt(d) for _ in range(L)]
a0 = rng.standard_normal(d)
a, writes = a0.copy(), []
for W in Ws:
    delta = np.tanh(rmsnorm(a) @ W)
    writes.append(delta)
    a = a + delta
assert np.abs(a - (a0 + np.sum(writes, axis=0))).max() < 1e-13
print("unrolling %d pre-norm blocks: a^(L) - a^(0) - sum of writes = %.1e"
      % (L, np.abs(a - (a0 + np.sum(writes, axis=0))).max()))

# and the assumption is architecture-conditional.  A post-norm trunk puts the
# normaliser ON the residual path, so no such decomposition exists.
b = a0.copy()
for W in Ws:
    b = rmsnorm(b + np.tanh(b @ W))
assert np.linalg.norm(b - (a0 + np.sum(writes, axis=0))) > 0.5 * np.linalg.norm(a)
print("the same weights in a post-norm trunk land %.2f away in norm: step 1 is "
      "false there, and with it the rest of the chapter"
      % np.linalg.norm(b - (a0 + np.sum(writes, axis=0))))

# Steps 4 to 6.  With a = a0 + sum c_j u_j, a linear read is affine in every
# intensity, and reading along u_i recovers c_i plus the overlap terms.
m_feat = 40
U = rng.standard_normal((m_feat, d))
U /= np.linalg.norm(U, axis=1, keepdims=True)
c = rng.standard_normal(m_feat) * 0.4
stream = a0 + c @ U
w = rng.standard_normal(d)
assert abs(stream @ w - (a0 @ w + float(c @ (U @ w)))) < 1e-12
i = 7
overlap = sum(c[j] * float(U[j] @ U[i]) for j in range(m_feat) if j != i)
assert abs(stream @ U[i] - (a0 @ U[i] + c[i] + overlap)) < 1e-12
assert abs(float(U[i] @ U[i]) - 1.0) < 1e-12         # the coefficient of c_i
print("reading along u_i returns c_i with coefficient exactly 1, plus a0's "
      "component and an interference term of %.4f from the other %d features: "
      "that last sum is the price of the whole scheme" % (overlap, m_feat - 1))

# Step 7, the falsification protocol.  Intervening a <- a + t u_i must give a
# response AFFINE in t, up to step 3's rescaling and the softmax's saturation.
readout = rng.standard_normal(d)
ts = np.linspace(-3, 3, 61)
resp = np.array([(stream + t * U[i]) @ readout for t in ts])
fit = np.polyfit(ts, resp, 1)
assert np.abs(resp - np.polyval(fit, ts)).max() < 1e-12
assert abs(fit[0] - float(U[i] @ readout)) < 1e-12
print("the logit response to a + t u_i is affine in t with slope <u_i, w> = "
      "%.4f, to %.1e" % (fit[0], np.abs(resp - np.polyval(fit, ts)).max()))

# Step 3, and the consequence that matters in practice.  RMSNorm discards the
# global scale, so a read sees c_i RELATIVE to ||a||: a fixed absolute steering
# coefficient is a shrinking relative one as the stream grows with depth.
for s in (0.5, 1.0, 7.0):
    assert np.abs(rmsnorm(s * stream) - rmsnorm(stream)).max() < 1e-12
norms = []
a = a0.copy()
for W in Ws:
    a = a + np.tanh(rmsnorm(a) @ W)
    norms.append(float(np.linalg.norm(a)))
assert norms[-1] > 1.5 * norms[0]
coeff = 2.0
rel = [coeff / n for n in norms]
assert rel[0] / rel[-1] > 1.5
print("the stream norm grows from %.2f at layer 1 to %.2f at layer %d, so a "
      "bare coefficient of %.1f is worth %.3f of the stream at the bottom and "
      "%.3f at the top: specify it as a fraction of rms(a)"
      % (norms[0], norms[-1], L, coeff, rel[0], rel[-1]))

# Step 8, the steering corollary.  A difference of conditional means is exactly
# the sum of the intensity differences along their own directions, so it is a
# single feature only when exactly one intensity moved.
def mean_stream(rng_, delta_c, n=4000):
    base = rng_.standard_normal((n, m_feat)) * 0.4
    return (a0 + (base + delta_c) @ U).mean(axis=0)


dc_one = np.zeros(m_feat)
dc_one[3] = 1.5
v_one = mean_stream(np.random.default_rng(1), dc_one) - mean_stream(
    np.random.default_rng(1), np.zeros(m_feat))
cos_one = float(v_one @ U[3] / (np.linalg.norm(v_one) * np.linalg.norm(U[3])))
dc_mix = np.zeros(m_feat)
dc_mix[[3, 11, 25]] = [1.5, -1.0, 0.8]
v_mix = mean_stream(np.random.default_rng(1), dc_mix) - mean_stream(
    np.random.default_rng(1), np.zeros(m_feat))
cos_mix = float(v_mix @ U[3] / (np.linalg.norm(v_mix) * np.linalg.norm(U[3])))
assert abs(cos_one - 1.0) < 1e-9
assert abs(cos_mix) < 0.85
assert np.abs(v_mix - (dc_mix @ U)).max() < 1e-10
print("with one intensity moved the difference of means is u_3 exactly "
      "(cosine %.6f); with three moved it is a mixture (cosine %.4f), and "
      "steering along it steers all three" % (cos_one, cos_mix))
'''

S2 = r'''
import math

from arith.model_d import MODEL_D
from arith.sae_capacity import capacity, eps_for

SEED = 1615
d = MODEL_D.d

# (16.9) and the one-line rank argument it is measured against.  Exact
# orthogonality gives d directions; almost-orthogonality gives exp(d eps^2/4).
assert abs(capacity(d, 0.1) - math.exp(d * 0.01 / 4)) < 1e-6
assert round(capacity(d, 0.1)) == 28001
assert round(capacity(d, 0.05)) == 13
assert abs(capacity(d, 0.1) / d - 6.84) < 0.01
print("at d = %d: exact orthogonality gives %d directions, (16.9) at eps = 0.1 "
      "gives %.0f, which is only %.1fx more"
      % (d, d, capacity(d, 0.1), capacity(d, 0.1) / d))

# The bound is VACUOUS over most of the range anyone would want to quote.
eps_vac = math.sqrt(4 * math.log(d) / d)
assert abs(eps_vac - 0.0901) < 1e-4
assert capacity(d, eps_vac) - d < 1e-6
for eps in (0.05, 0.08, 0.090, 0.10, 0.15):
    print("  eps = %.3f: permits %12.3e, that is %.3g x d"
          % (eps, capacity(d, eps), capacity(d, eps) / d))
assert capacity(d, 0.08) < d and capacity(d, 0.095) > d
print("below eps = %.4f the bound permits FEWER than d directions, so it is "
      "not loose there, it is vacuous" % eps_vac)

# and the exponent is quadratic in eps, so "inside the bound" is decided by the
# third significant figure of a threshold nobody can justify.
d_eps = 2 / (d * 0.1)
assert abs(d_eps - 0.00488) < 1e-5
assert abs(math.log(capacity(d, 0.1 + d_eps) / capacity(d, 0.1)) - 1.0) < 0.03
m_sae = 32 * d
assert m_sae == 131_072
assert abs(eps_for(d, m_sae) - 0.10727) < 1e-5
assert abs(capacity(d, eps_for(d, m_sae)) / m_sae - 1.0) < 1e-9
assert abs(m_sae / capacity(d, 0.1) - 4.68) < 0.01
print("an expansion-32 dictionary of %d features is admitted at eps = %.6f and "
      "sits %.2fx outside the bound at eps = 0.100; the permitted count moves "
      "by a factor of e for a change of %.4f in eps"
      % (m_sae, eps_for(d, m_sae), m_sae / capacity(d, 0.1), d_eps))

# THE HONESTY STEP.  What is proved is existence and the eps^2 d scaling of the
# exponent.  What is not proved is that the true maximum is near the bound, and
# the union bound over m^2 nearly independent events is crude.  Measure it.
def random_dictionary(m, dd, rng):
    U = rng.standard_normal((m, dd))
    return U / np.linalg.norm(U, axis=1, keepdims=True)


def max_coherence(U):
    G = U @ U.T
    np.fill_diagonal(G, 0.0)
    return float(np.abs(G).max())


def typical_coherence(m, dd, trials=5):
    """Median over `trials` independent draws, so the predicate below is a
    deterministic function of m and the bisection is well defined."""
    return float(np.median([
        max_coherence(random_dictionary(
            m, dd, np.random.default_rng((SEED, dd, m, t))))
        for t in range(trials)]))


def largest_m_within(eps, dd, hi=4096):
    lo, best = 2, 2
    while lo <= hi:
        mid = (lo + hi) // 2
        if typical_coherence(mid, dd) <= eps:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


ratios = []
for dd, eps in ((128, 0.30), (256, 0.25), (512, 0.20)):
    m = largest_m_within(eps, dd)
    bound = math.exp(dd * eps ** 2 / 4)
    ratios.append(m / bound)
    assert m >= bound, (dd, m, bound)          # a lower bound on the achievable
    assert 2.0 < m / bound < 8.0, (dd, m / bound)
    print("  d = %3d, eps = %.2f: achieved m = %4d against exp(d eps^2/4) = "
          "%7.1f, a factor of %.2f" % (dd, eps, m, bound, m / bound))
assert max(ratios) < 10.0                      # SINGLE DIGITS, not 10^2 to 10^3
assert max(ratios) / min(ratios) < 3.0         # and it grows slowly with d
print("the union bound is loose by %.1f to %.1f, which is single digits and "
      "growing slowly with d: it is far more respectable than folklore's two "
      "or three orders of magnitude" % (min(ratios), max(ratios)))
'''

S3 = r'''
import math

from scipy.stats import norm

from arith.model_d import MODEL_D
from arith.sae_capacity import (capacity, random_sign_k, sustainable_l0,
                                worst_case_k)

SEED = 1603
rng = np.random.default_rng(SEED)
d, m = MODEL_D.d, 2 * MODEL_D.d

# Step 3, the worst case, which mentions neither d nor m and is brutal.
assert abs(worst_case_k(0.1, 1.0) - 5.5) < 1e-9
assert abs(worst_case_k(0.1, 1.0) - 0.5 * (1 + 1 / 0.1)) < 1e-12
# Step 5, the refinement: independent interference signs buy a SQUARE.
assert abs(random_sign_k(0.1, 1.0) - 100.0) < 1e-9
assert random_sign_k(0.1) / worst_case_k(0.1) > 18
print("at eps = 0.1: worst-case superposition supports k < %.1f live features; "
      "with independent signs, k ~ %.0f. The square is the whole reason "
      "superposition is usable." % (worst_case_k(0.1), random_sign_k(0.1)))

# Step 6, the sharper route, and the one that predicts the measurement.  For a
# random unit dictionary E[<u_i,u_j>^2] = 1/d exactly, so the interference at an
# ACTIVE feature has variance (k-1)/d and an inactive read has variance k/d.
U = rng.standard_normal((m, d)).astype(np.float32)
U /= np.linalg.norm(U, axis=1, keepdims=True)
G = (U[:400] @ U[:400].T).astype(np.float64)
np.fill_diagonal(G, 0.0)
off = G[~np.eye(400, dtype=bool)]
assert abs((off ** 2).mean() * d - 1.0) < 0.05
print("E[<u_i,u_j>^2] measured over %d pairs is %.3e against 1/d = %.3e"
      % (off.size, (off ** 2).mean(), 1 / d))

stats = {}
for k in (10, 50, 200):
    act, ina = [], []
    for _ in range(200):
        S = rng.choice(m, k, replace=False)
        r = (U @ U[S].sum(axis=0)).astype(np.float64)
        mask = np.ones(m, dtype=bool)
        mask[S] = False
        act.append(r[S] - 1.0)
        ina.append(r[mask])
    a = np.concatenate(act)
    b = np.concatenate(ina)
    stats[k] = (float(a.std()), float(b.std()))
    assert abs(a.std() / math.sqrt((k - 1) / d) - 1.0) < 0.12, (k, a.std())
    assert abs(b.std() / math.sqrt(k / d) - 1.0) < 0.05, (k, b.std())
    print("  k=%3d: interference sd at active features %.4f (sqrt((k-1)/d) = "
          "%.4f), at inactive %.4f (sqrt(k/d) = %.4f), separation %.2f sigma"
          % (k, a.std(), math.sqrt((k - 1) / d), b.std(), math.sqrt(k / d),
             1 / b.std()))

# The correction the chapter makes.  At d = 4096 and k = 200 the separation is
# still about 4.5 standard deviations, so the reads are trivially separable and
# a draft claim of an AUC below 0.75 there is off by an enormous margin.
sep200 = 1 / stats[200][1]
assert abs(stats[200][1] - 0.222) < 0.004
assert abs(sep200 - 4.5) < 0.1
assert norm.cdf(sep200) > 0.999
k_auc75 = d / norm.ppf(0.75) ** 2
assert abs(k_auc75 - 9003) < 5
assert k_auc75 > 2 * d
print("at k = 200 the separation is %.2f sigma and the AUC is %.6f; reaching "
      "an AUC of 0.75 needs k = %.0f, which is more than twice d"
      % (sep200, norm.cdf(sep200), k_auc75))
for k in (10, 50, 200):
    assert abs(1 / stats[k][1] - math.sqrt(d / k)) / math.sqrt(d / k) < 0.05
print("the separation falls as sqrt(d/k): %s standard deviations at k = 10, "
      "50, 200" % ", ".join("%.1f" % (1 / stats[k][1]) for k in (10, 50, 200)))

# Steps 7 and 8.  Surviving m simultaneous reads costs z^2 = 4 ln m standard
# deviations, and k <= d/z^2 is then (16.14) with no elimination step at all.
z2 = 4 * math.log(m)
assert abs(d / z2 - sustainable_l0(d, m)) < 1e-9
for width in (8, 16, 32, 64):
    mm = width * d
    k_max = sustainable_l0(d, mm)
    assert abs(k_max - d / (4 * math.log(mm))) < 1e-9
    assert abs(math.exp(d / (4 * k_max)) / mm - 1.0) < 1e-6   # exp(d/4k) = m
    print("  expansion %3d, m = %7d: sustainable L0 = %.1f" % (width, mm, k_max))
assert abs(sustainable_l0(d, 131_072) - 86.9) < 0.1
assert abs(sustainable_l0(d, 32_768) - 98.5) < 0.1
assert sustainable_l0(d, 131_072) < sustainable_l0(d, 32_768)
print("(16.14) sets l0_target = d // (4 ln m): %d at m = 131,072 and %d at "
      "m = 32,768, one configuration line rather than a sweep"
      % (round(sustainable_l0(d, 131_072)), round(sustainable_l0(d, 32_768))))

# Read it off.  Capacity is exponential in d/k, the width per ACTIVE feature,
# and not in d: doubling the live count halves the exponent.
for k in (10, 20, 40):
    assert abs(math.log(capacity(d, math.sqrt(1 / k))) - d / (4 * k)) < 1e-6
assert abs(math.log(capacity(d, math.sqrt(1 / 20)))
           - 0.5 * math.log(capacity(d, math.sqrt(1 / 10)))) < 1e-6
print("eliminating eps against (16.9) gives ln m <= d/4k, so doubling the "
      "live-feature count halves the exponent: that is the actual content of "
      "'superposition works because features are sparse'")
'''

S4 = r'''
from arith.model_d import MODEL_D
from arith.sae_capacity import layer_params, sae_params, soft_threshold_deficit

SEED = 1604
rng = np.random.default_rng(SEED)
d = MODEL_D.d


def soft_threshold(c, lam):
    """(16.17).  The threshold is lam/2, not lam: the factor of two comes from
    differentiating the squared reconstruction term."""
    return np.sign(c) * np.maximum(np.abs(c) - lam / 2.0, 0.0)


# Steps 2 to 6, against a numerical minimisation of the objective itself, on an
# orthonormal active set so that the stationarity condition is exact.
k_active = 6
Q, _ = np.linalg.qr(rng.standard_normal((24, k_active)))
W = Q.T                                              # rows are the atoms
assert np.abs(W @ W.T - np.eye(k_active)).max() < 1e-12
x = rng.standard_normal(24) * 1.2
c = W @ x
for lam in (0.0, 0.2, 0.8, 2.0):
    z = soft_threshold(c, lam)
    obj = lambda v: float(((x - v @ W) ** 2).sum() + lam * np.abs(v).sum())
    base = obj(z)
    for _ in range(500):                             # nothing nearby is better
        pert = z + rng.standard_normal(k_active) * 0.05
        assert obj(pert) >= base - 1e-12, lam
    active = np.abs(c) > lam / 2
    assert np.abs(z[~active]).max(initial=0.0) == 0.0        # exact zeros
    assert np.abs(np.abs(z[active]) - (np.abs(c[active]) - lam / 2)).max(
        initial=0.0) < 1e-12
    assert (np.sign(z[active]) == np.sign(c[active])).all()
    print("lambda = %.1f: %d of %d atoms survive, each short by exactly "
          "%.3f = lambda/2" % (lam, active.sum(), k_active, lam / 2))

# Step 7, the two consequences, because ONE lambda does both.  Selection is
# wanted; shrinkage is not, and it cannot be tuned away separately.
lam = 0.4
for n_active in (2, 8, 32, 128, 512):
    cbar = 1.0
    ca = np.full(n_active, cbar)
    za = soft_threshold(ca, lam)
    ratio = float(np.linalg.norm(za) / np.linalg.norm(ca))
    # (16.18): INDEPENDENT of how many atoms are active
    assert abs(ratio - (1 - lam / (2 * cbar))) < 1e-12
    assert abs(ratio - soft_threshold_deficit(lam, cbar)) < 1e-12
    assert abs(float(np.abs(ca - za).mean()) - lam / 2) < 1e-12
print("with lambda/2 = %.1f and cbar = 1, the reconstruction is %.0f%% short "
      "at every active count from 2 to 512, and the per-coefficient bias is "
      "exactly lambda/2 = %.2f" % (lam / 2, 100 * lam / 2, lam / 2))
for lam2 in (0.1, 0.4, 1.0):
    assert abs(soft_threshold_deficit(lam2, 1.0) - (1 - lam2 / 2)) < 1e-12
    print("  lambda = %.1f -> reconstruction norm ratio %.2f"
          % (lam2, soft_threshold_deficit(lam2, 1.0)))

# and on a realistic spread of magnitudes the bias is still exactly lambda/2 per
# surviving coefficient, which is what makes it a property of lambda alone.
mags = np.abs(rng.lognormal(0.0, 0.6, 4000)) + 0.3
zz = soft_threshold(mags, lam)
surv = zz > 0
assert np.abs((mags[surv] - zz[surv]) - lam / 2).max() < 1e-12
assert surv.all()
# with a spread of magnitudes the right cbar is the second moment over the
# first, and (16.18) still holds to better than a fifth of a per cent
cstar = float((mags ** 2).sum() / mags.sum())
ratio = float(np.linalg.norm(zz) / np.linalg.norm(mags))
assert abs(ratio - (1 - lam / (2 * cstar))) < 0.002
assert ratio < 1 - lam / (2 * mags.max())
print("on %d coefficients of mixed magnitude every survivor is short by "
      "exactly %.3f, and the norm ratio %.4f matches 1 - lambda/(2 cbar) = "
      "%.4f at cbar = %.4f" % (mags.size, lam / 2, ratio,
                               1 - lam / (2 * cstar), cstar))

# Step 9.  TopK deletes step 3, so stationarity gives z_j = c_j on the chosen
# set and there is NO shrinkage.  The cost is that the sparsity is fixed.
def topk(c, k):
    out = np.zeros_like(c)
    idx = np.argsort(-np.abs(c))[:k]
    out[idx] = c[idx]
    return out


for k in (2, 4, 6):
    t = topk(c, k)
    nz = t != 0
    assert np.abs(t[nz] - c[nz]).max() == 0.0        # unchanged, not shrunk
    assert nz.sum() == k                             # and always exactly k
assert np.abs(topk(c, k_active) - c).max() == 0.0
print("TopK returns the chosen coefficients unchanged, so there is no "
      "shrinkage to correct; the cost is that |A| = k whatever the token needs")

# Step 11, the load-bearing constraint.  Without unit-norm decoder rows the
# penalty is gameable: scale an atom up and its coefficient scales down, so the
# L1 term falls with no change to the reconstruction at all.
w1 = rng.standard_normal(24)
w1 /= np.linalg.norm(w1)
coef = 0.9
for s in (1.0, 2.0, 10.0):
    assert np.abs((coef / s) * (s * w1) - coef * w1).max() < 1e-12
    assert abs(np.abs(coef / s) - np.abs(coef) / s) < 1e-15
print("scaling a decoder row by 10 and its coefficient by 1/10 leaves the "
      "reconstruction identical and divides the L1 penalty by 10: without "
      "||w_j|| = 1 the sparsity term measures nothing")

# A-16.1.  What the dictionary costs, which is the chapter's third number.
p = sae_params(d, 32)
lay = layer_params(MODEL_D)
assert p["m"] == 131_072
assert p["W_enc"] == p["W_dec"] == d * p["m"]
assert p["total"] == 2 * d * p["m"] + p["m"] + d
assert lay == 2 * d * d + 2 * d * MODEL_D.n_kv * MODEL_D.d_h + 3 * d * MODEL_D.d_ff
assert abs(p["total"] / lay - 4.92) < 0.01
print("a Model D sparse autoencoder at expansion 32: m = %d, %.3f B "
      "parameters, %.2fx the %.1f M-parameter layer it explains"
      % (p["m"], p["total"] / 1e9, p["total"] / lay, lay / 1e6))
print("one per layer, %d layers: %.1f B parameters, %.0f GiB of bf16 weights"
      % (MODEL_D.L, MODEL_D.L * p["total"] / 1e9,
         MODEL_D.L * 2 * (p["W_enc"] + p["W_dec"]) / (1 << 30)))
'''

SECTIONS = [
    ("1", "Additivity of the residual stream",
     "A pre-norm block adds its output to the stream, so unrolling the trunk is "
     "an identity and the final activation is the embedding plus a sum of "
     "writes. If features are written as fixed directions scaled by their "
     "intensities, every linear read is affine in every intensity, and reading "
     "along one feature returns its own intensity plus the overlaps with the "
     "others. The cell checks all of that, then checks the two things that "
     "make the hypothesis architecture-conditional: a post-norm trunk breaks "
     "the first step outright, and the normaliser means a steering coefficient "
     "is relative to the layer's own scale.",
     S1),
    ("2", "Almost-orthogonal capacity",
     "Concentration on the sphere plus a union bound over pairs gives a count "
     "of almost-orthogonal directions exponential in d at fixed coherence. The "
     "cell evaluates it at Model D's width, where it is vacuous below a "
     "coherence of 0.09 and permits only about seven times d at 0.1, and shows "
     "that the permitted count moves by a factor of e for a change of five "
     "thousandths in the threshold. It closes by measuring how loose the union "
     "bound actually is, by bisection on random dictionaries, and the answer is "
     "single digits.",
     S2),
    ("3", "Interference, and the sparsity condition",
     "A matched-filter read of a superposed vector returns the intensity plus a "
     "sum of overlaps, and bounding every overlap by the coherence gives a "
     "worst case of about five live features that mentions neither the width "
     "nor the dictionary size. Modelling the signs as independent buys a square "
     "and takes that to about a hundred; working with the variance instead is "
     "sharper still, and predicts the measured interference standard deviation "
     "to three decimals. Eliminating the coherence gives the live-feature "
     "limit, which is the only prediction in the chapter.",
     S3),
    ("4", "L1 shrinkage in the sparse-autoencoder objective",
     "Stationarity of a squared reconstruction plus an L1 penalty on an "
     "orthonormal active set is the soft threshold, with the threshold at half "
     "lambda rather than lambda. One coefficient does two jobs: it selects, "
     "which is wanted, and it shrinks every survivor by exactly half lambda, "
     "which is not and which no sparsity sweep can remove. The cell asserts "
     "that independence explicitly, shows why TopK has no shrinkage and why "
     "the unit-norm decoder constraint is load-bearing, and ends with the bill.",
     S4),
]
