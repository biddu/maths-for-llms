"""Chapter 5 — Normalisation and the Residual Stream.

Generated into `notebooks/ch05_norm_residual.ipynb` by `build_all.py`.  The
section numbers are a contract: the chapter's margin notes point at §2, §3, §4
and §5 by number, so they may be added to but never renumbered.
"""
from __future__ import annotations

CHAPTER = 5
SLUG = "norm_residual"
TITLE = "Normalisation and the Residual Stream"
BLURB = (
    "Both normalisation Jacobians, checked against central differences, and the "
    "one-rank gap between them. Then what the residual wiring does to a product "
    "of Jacobians, and the logit bound QK-norm buys for 8192 parameters."
)

# ---------------------------------------------------------------------------
S1 = r'''
from arith.model_d import MODEL_D, norm_stats, non_embedding

SEED = 5001
rng = np.random.default_rng(SEED)
c = MODEL_D
s = norm_stats(c)

# What removing centring costs, in parameters.  Nothing here is typed from the
# page: norm_stats owns the counts and this cell only checks the arithmetic.
assert s["sites"] == 2 * c.L + 1 == 65
assert s["rmsnorm_params"] == s["sites"] * c.d == 266_240
assert s["layernorm_params"] == 2 * s["rmsnorm_params"] == 532_480
saving = s["layernorm_params"] - s["rmsnorm_params"]
assert saving == s["rmsnorm_params"]
assert round(100 * saving / non_embedding(c), 4) == 0.0038
print("sites %d, RMSNorm %d params, LayerNorm %d, saving %.4f%% of the model"
      % (s["sites"], s["rmsnorm_params"], s["layernorm_params"],
         100 * saving / non_embedding(c)))

# Equation (5.3): what centring removes from the SQUARED norm is exactly rho^2.
# The activations are synthetic, as F-5.1 is; the identity is not.
d = c.d
C = np.eye(d) - np.ones((d, d)) / d
for trial in range(8):
    x = rng.standard_normal(d) + 0.012 * rng.standard_normal()  # small offset
    rho = abs(x.mean()) / np.sqrt(np.mean(x * x))
    lost = (x @ x - (x @ C) @ (x @ C)) / (x @ x)
    assert abs(lost - rho ** 2) < 1e-12, (trial, lost, rho ** 2)

# and the claim the box states so it can fail: rho is O(1e-2), so rho^2 is the
# part in ten thousand that centring deletes.
rhos = np.array([abs(v.mean()) / np.sqrt(np.mean(v * v))
                 for v in (rng.standard_normal((256, d)) + 0.012)])
assert 1e-3 < np.median(rhos) < 1e-1, np.median(rhos)
print("median rho %.4f, so centring deletes %.2e of the squared norm"
      % (np.median(rhos), np.median(rhos) ** 2))
'''

S2 = r'''
SEED = 5002
rng = np.random.default_rng(SEED)
d = 24
g = 1.0 + 0.2 * rng.standard_normal(d)
x = 1.7 * rng.standard_normal(d) + 0.4


def rms(v, eps=0.0):
    return np.sqrt(np.mean(v * v) + eps)


def rmsnorm(v, gain, eps=0.0):
    return v / rms(v, eps) * gain


# D-5.1 in the book's denominator layout, so that entry [j, i] is dy_i/dx_j and
# the backward form needs no transpose.
r = rms(x)
xhat = x / np.linalg.norm(x)
P = np.eye(d) - np.outer(xhat, xhat)
J = (P @ np.diag(g)) / r

# central differences, which is the only check that catches a wrong layout
h = 1e-6
Jfd = np.empty((d, d))
for j in range(d):
    e = np.zeros(d); e[j] = h
    Jfd[j] = (rmsnorm(x + e, g) - rmsnorm(x - e, g)) / (2 * h)
err = np.abs(J - Jfd).max()
assert err < 1e-8, err
print("RMSNorm Jacobian vs central difference: max abs %.2e" % err)

# The projector, step 6.  Symmetric, idempotent, rank d-1, null space span(x).
assert np.abs(P - P.T).max() < 1e-14
assert np.abs(P @ P - P).max() < 1e-14
assert np.linalg.matrix_rank(P, tol=1e-10) == d - 1
assert np.abs(P @ x).max() < 1e-12                      # radial direction killed
assert abs(x @ J).max() < 1e-12                         # step 7's corollary

# Scale invariance, and the 1/RMS that comes with it.
assert np.abs(rmsnorm(3.7 * x, g) - rmsnorm(x, g)).max() < 1e-13
ybar = rng.standard_normal(d)
xbar = P @ (g * ybar) / r
assert np.abs(xbar - Jfd @ ybar).max() < 1e-8
assert np.abs(P @ (g * ybar) / rms(4 * x) - 0.25 * xbar).max() < 1e-13

# The failure mode: with eps inside the root the map is not a projector, and
# the corrected Jacobian is the one that matches finite differences.
eps = 1e-2                      # exaggerated, so the discrepancy is visible
re = rms(x, eps)
Jeps = (np.eye(d) - np.outer(xhat, xhat) * (re ** 2 - eps) / re ** 2) @ np.diag(g) / re
Jefd = np.empty((d, d))
for j in range(d):
    e = np.zeros(d); e[j] = h
    Jefd[j] = (rmsnorm(x + e, g, eps) - rmsnorm(x - e, g, eps)) / (2 * h)
assert np.abs(Jeps - Jefd).max() < 1e-8
naive = (P @ np.diag(g)) / re
assert np.abs(naive - Jefd).max() > 1e-4, "the eps-free form should be visibly wrong"
print("with eps inside the root: corrected %.2e, eps-free form %.2e"
      % (np.abs(Jeps - Jefd).max(), np.abs(naive - Jefd).max()))
'''

S3 = r'''
SEED = 5003
rng = np.random.default_rng(SEED)
d = 24
g = 1.0 + 0.2 * rng.standard_normal(d)
x = 1.7 * rng.standard_normal(d) + 0.9          # deliberately off-centre
one = np.ones(d)
C = np.eye(d) - np.outer(one, one) / d


def layernorm(v, gain):
    z = v @ C
    return z / np.sqrt(np.mean(z * z)) * gain


def rmsnorm(v, gain):
    return v / np.sqrt(np.mean(v * v)) * gain


# D-5.2, same layout as D-5.1.
z = x @ C
sigma = np.linalg.norm(z) / np.sqrt(d)
zhat = z / np.linalg.norm(z)
Pln = C - np.outer(zhat, zhat)
Jln = (Pln @ np.diag(g)) / sigma

h = 1e-6
Jfd = np.empty((d, d))
for j in range(d):
    e = np.zeros(d); e[j] = h
    Jfd[j] = (layernorm(x + e, g) - layernorm(x - e, g)) / (2 * h)
assert np.abs(Jln - Jfd).max() < 1e-8, np.abs(Jln - Jfd).max()
print("LayerNorm Jacobian vs central difference: max abs %.2e"
      % np.abs(Jln - Jfd).max())

# Step 4: z is centred, so C zhat = zhat and C(I - zhat zhat^T) = C - zhat zhat^T.
assert abs(one @ z) < 1e-12
assert np.abs(C @ zhat - zhat).max() < 1e-13
assert np.abs(C @ (np.eye(d) - np.outer(zhat, zhat)) - Pln).max() < 1e-13

# Step 5: still an orthogonal projector, but of rank d-2.
xhat = x / np.linalg.norm(x)
Prms = np.eye(d) - np.outer(xhat, xhat)
assert np.abs(Pln - Pln.T).max() < 1e-14 and np.abs(Pln @ Pln - Pln).max() < 1e-13
assert np.linalg.matrix_rank(Pln, tol=1e-10) == d - 2
assert np.linalg.matrix_rank(Prms, tol=1e-10) == d - 1

# The extra killed direction, which is the whole content of "minus the mean".
assert np.abs(Pln @ one).max() < 1e-12 and np.abs(Pln @ z).max() < 1e-12
assert np.abs(Prms @ x).max() < 1e-12                   # RMSNorm kills one
assert np.linalg.norm(Prms @ one) > 1.0                 # but not the all-ones
sv_ln = np.linalg.svd(Pln, compute_uv=False)
sv_rms = np.linalg.svd(Prms, compute_uv=False)
assert sum(sv_ln < 1e-10) == 2 and sum(sv_rms < 1e-10) == 1
assert np.abs(sv_ln[:d - 2] - 1).max() < 1e-12          # a projector, not a scaling
print("null dimensions: LayerNorm %d, RMSNorm %d, so LayerNorm kills one more"
      % (sum(sv_ln < 1e-10), sum(sv_rms < 1e-10)))

# Both Jacobians are a positive scalar times an orthogonal projector, so with
# unit gain neither can amplify a gradient.
for J, scale in ((Pln / sigma, 1 / sigma), (Prms / np.sqrt(np.mean(x * x)), 1)):
    assert np.linalg.svd(J, compute_uv=False)[0] <= scale + 1e-12
'''

S4 = r'''
SEED = 5004
rng = np.random.default_rng(SEED)
d, L = 16, 32


def rmsnorm(v):
    return v / np.sqrt(np.mean(v * v))


def jac_norm(v):                        # D-5.1 with unit gain, layout [j, i]
    vh = v / np.linalg.norm(v)
    return (np.eye(d) - np.outer(vh, vh)) / np.sqrt(np.mean(v * v))


# scaled so that the per-layer delta lands near 1/L, which is the regime
# D-7.4 step 3 identifies as the one where depth is free
W1 = [0.11 * rng.standard_normal((d, d)) / np.sqrt(d) for _ in range(L)]
W2 = [0.11 * rng.standard_normal((d, d)) / np.sqrt(d) for _ in range(L)]


def F(u, l):
    return np.tanh(u @ W1[l]) @ W2[l]


def jac_F(u, l):
    t = np.tanh(u @ W1[l])
    return W1[l] @ np.diag(1 - t * t) @ W2[l]


x0 = rng.standard_normal(d) * 1.4

# Step 1 and step 2, against central differences on one block.
h = 1e-6
for wiring in ("pre", "post"):
    if wiring == "pre":
        block = lambda v: v + F(rmsnorm(v), 0)
        Jan = np.eye(d) + jac_norm(x0) @ jac_F(rmsnorm(x0), 0)
    else:
        block = lambda v: rmsnorm(v + F(v, 0))
        y0 = x0 + F(x0, 0)
        Jan = (np.eye(d) + jac_F(x0, 0)) @ jac_norm(y0)
    Jfd = np.empty((d, d))
    for j in range(d):
        e = np.zeros(d); e[j] = h
        Jfd[j] = (block(x0 + e) - block(x0 - e)) / (2 * h)
    assert np.abs(Jan - Jfd).max() < 1e-7, (wiring, np.abs(Jan - Jfd).max())
    print("%s-norm block Jacobian vs central difference: %.2e"
          % (wiring, np.abs(Jan - Jfd).max()))

# Now the product over L layers, in both wirings.
prod_pre, prod_post = np.eye(d), np.eye(d)
xp, xq, deltas = x0.copy(), x0.copy(), []
for l in range(L):
    Jl = jac_norm(xp) @ jac_F(rmsnorm(xp), l)
    deltas.append(np.linalg.svd(Jl, compute_uv=False)[0])
    prod_pre = prod_pre @ (np.eye(d) + Jl)
    xp = xp + F(rmsnorm(xp), l)
    yq = xq + F(xq, l)
    prod_post = prod_post @ ((np.eye(d) + jac_F(xq, l)) @ jac_norm(yq))
    xq = rmsnorm(yq)
delta = max(deltas)

# Step 3: the identity term is exact, appears once, and does not depend on L.
# The drift bound is D-7.4's, previewed here because it is what bounds it.
drift = np.linalg.svd(prod_pre - np.eye(d), compute_uv=False)[0]
assert drift <= (1 + delta) ** L - 1
assert delta * L < 2.0, "keep the blocks in the delta = O(1/L) regime"
smin_pre = np.linalg.svd(prod_pre, compute_uv=False)[-1]
assert smin_pre > 0.5, smin_pre

# Step 4: post-norm puts a projector ON the path, so the product is rank
# deficient.  There is a direction the gradient cannot reach x_0 along at all,
# and no L and no J_F can repair it: that is the structural difference.
smin_post = np.linalg.svd(prod_post, compute_uv=False)[-1]
assert smin_post < 1e-10, smin_post
assert np.linalg.matrix_rank(prod_post, tol=1e-8) <= d - 1
print("delta %.4f, pre-norm drift %.4f (bound %.4f), sigma_min pre %.3f post %.2e"
      % (delta, drift, (1 + delta) ** L - 1, smin_pre, smin_post))

# Step 6, the cost the standard argument omits.  With uncorrelated zero-mean
# writes the variances add, so the stream norm grows like sqrt(l).
trials, a = 4000, 0.6
starts = rng.standard_normal((trials, d))
acc = starts.copy()
for l in range(L):
    acc = acc + a * rng.standard_normal((trials, d)) / np.sqrt(d) * np.sqrt(d)
mean_sq = np.mean(np.sum(acc * acc, axis=1))
predicted = d + L * a * a * d
assert abs(mean_sq / predicted - 1) < 0.02, (mean_sq, predicted)
print("stream growth: measured E||x_L||^2 %.1f against the idealisation %.1f, "
      "a factor %.2f in norm" % (mean_sq, predicted, np.sqrt(mean_sq / d)))
'''

S5 = r'''
from arith.model_d import MODEL_D, norm_stats

SEED = 5005
rng = np.random.default_rng(SEED)
c = MODEL_D
d_h = c.d_h
bound = np.sqrt(d_h)

# The parameter cost, from arith/, and the bound it buys.
assert norm_stats(c)["qk_norm_params"] == 2 * d_h * c.L == 8192
assert round(bound, 2) == 11.31
print("qk_norm costs %d parameters and bounds every pre-gain logit at %.2f"
      % (norm_stats(c)["qk_norm_params"], bound))


def rmsnorm(v):
    return v / np.sqrt(np.mean(v * v, axis=-1, keepdims=True))


# Step 1: unit RMS on a d_h-vector means norm exactly sqrt(d_h).
q = rmsnorm(rng.standard_normal((4096, d_h)) * rng.uniform(0.1, 30, (4096, 1)))
k = rmsnorm(rng.standard_normal((4096, d_h)) * rng.uniform(0.1, 30, (4096, 1)))
assert np.abs(np.linalg.norm(q, axis=1) - bound).max() < 1e-11

# Steps 2 and 3: Cauchy-Schwarz, then divide.  The bound holds for every pair,
# and the logit is sqrt(d_h) cos(theta) exactly.
logits = (q * k).sum(1) / bound
assert np.abs(logits).max() <= bound + 1e-12
cos = (q * k).sum(1) / (np.linalg.norm(q, axis=1) * np.linalg.norm(k, axis=1))
assert np.abs(logits - bound * cos).max() < 1e-12
print("max |logit| over %d normalised pairs: %.3f, against the bound %.3f"
      % (len(logits), np.abs(logits).max(), bound))

# The bound is tight, attained when k is parallel to q, and it is a real bound
# rather than a statistical one: without the normalisation there is none.
assert abs((q[0] @ q[0]) / bound - bound) < 1e-10
raw_q = rng.standard_normal((4096, d_h)) * 6.0
raw_k = raw_q * 1.3 + 0.1 * rng.standard_normal((4096, d_h))
assert (np.abs((raw_q * raw_k).sum(1) / bound) > bound).mean() > 0.9

# The failure mode: the bound also caps how peaked the attention can be.
ratio = np.exp(2 * bound)
assert round(ratio / 1e9, 2) == 6.71
assert round(np.exp(2 * np.sqrt(32)) / 1e4, 1) == 8.2
assert np.exp(2 * bound) > 1e9 > np.exp(2 * np.sqrt(32))
print("largest achievable weight ratio: %.2e at d_h = %d, %.2e at d_h = 32"
      % (ratio, d_h, np.exp(2 * np.sqrt(32))))
'''

SECTIONS = [
    ("1", "What centring is worth",
     "Centring removes the component of an activation along the all-ones "
     "direction, and equation (5.3) says exactly what that costs: the share of "
     "the squared norm removed is rho squared, where rho is the mean over the "
     "RMS. The cell checks that identity on synthetic activations (Figure 5.1 "
     "is synthetic too, and the chapter says so), and recomputes the parameter "
     "saving from arith/model_d.py rather than trusting the page.",
     S1),
    ("2", "The RMSNorm Jacobian",
     "D-5.1 says the Jacobian is one over the RMS times an orthogonal "
     "projector onto the complement of x, followed by the gain. The only "
     "honest check of a Jacobian is a central difference, so that is what this "
     "cell does, and it then verifies the three properties the projector is "
     "claimed to have (symmetric, idempotent, rank d minus one). The last "
     "block exercises the failure-mode note: with epsilon inside the root the "
     "map is no longer a projector, and the corrected form is the one that "
     "matches.",
     S2),
    ("3", "The LayerNorm Jacobian, and the extra killed direction",
     "D-5.2 chains the RMSNorm Jacobian through the centring matrix and gets a "
     "projector again, but onto the complement of the span of the all-ones "
     "vector and z, which is one dimension smaller. The cell checks the "
     "Jacobian against central differences and then counts null directions "
     "both ways, which is the entire mathematical content of the phrase "
     "LayerNorm minus the mean.",
     S3),
    ("4", "Why pre-norm removes the warmup requirement",
     "D-5.3 differentiates both wirings and finds one structural difference: "
     "pre-norm leaves an identity term that nothing multiplies, post-norm puts "
     "the normalisation Jacobian on the path. The cell verifies both block "
     "Jacobians against central differences, then takes the product over "
     "thirty-two layers and shows that the post-norm product is rank deficient "
     "(the projector on the path deletes a direction the gradient can never "
     "recover) while the pre-norm product stays close to the identity. The "
     "last block is step 6, the cost the usual telling omits.",
     S4),
    ("5", "The QK-norm logit bound",
     "Normalising q and k inside each head fixes both norms at the square root "
     "of the head dimension, so Cauchy-Schwarz turns a statistical argument "
     "into an actual bound. The cell checks the bound on several thousand "
     "normalised pairs, shows it is attained rather than loose, shows that "
     "unnormalised pairs violate it routinely, and prices the peakedness cap "
     "the bound implies at two head dimensions.",
     S5),
]
