"""Chapter 7 — Backpropagation Through a Transformer Block.

Generated into `notebooks/ch07_backprop.ipynb` by `build_all.py`.  The chapter
cites §1, §2, §4 and §6 by number, so sections may be added but never
renumbered.  Float64 throughout, at reduced shapes, with every analytic
gradient checked against a central difference.
"""
from __future__ import annotations

CHAPTER = 7
SLUG = "backprop"
TITLE = "Backpropagation Through a Transformer Block"
BLURB = (
    "Every backward formula in the chapter, at reduced Model D shapes in "
    "float64, checked against central differences. No autograd anywhere: the "
    "point is that the formulas are right, not that a framework agrees."
)

# ---------------------------------------------------------------------------
# A finite-difference helper is repeated in each section deliberately: a
# notebook cell that depends on an earlier cell's definitions is a cell that
# cannot be read on its own.
FD = r'''
def fd_grad(f, X, h=1e-6):
    """Central difference of a scalar-valued f at the array X."""
    X = np.asarray(X, dtype=float)
    g = np.zeros_like(X)
    it = np.nditer(X, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = X[i]
        X[i] = old + h; hi = f(X)
        X[i] = old - h; lo = f(X)
        X[i] = old
        g[i] = (hi - lo) / (2 * h)
        it.iternext()
    return g
'''

S1 = FD + r'''
SEED = 7001
rng = np.random.default_rng(SEED)
s, d_in, d_out = 6, 5, 4

A = rng.standard_normal((s, d_in))
W = rng.standard_normal((d_in, d_out))
G = rng.standard_normal((s, d_out))        # the upstream gradient, held fixed

loss = lambda Y: float((G * Y).sum())

# D-7.1.  Both products are forced by the shapes, which is the mnemonic.
W_bar = A.T @ G
A_bar = G @ W.T
assert W_bar.shape == W.shape and A_bar.shape == A.shape
assert np.abs(W_bar - fd_grad(lambda Wv: loss(A @ Wv), W)).max() < 1e-8
assert np.abs(A_bar - fd_grad(lambda Av: loss(Av @ W), A)).max() < 1e-8
print("dW and dA against central differences: %.2e, %.2e"
      % (np.abs(W_bar - fd_grad(lambda Wv: loss(A @ Wv), W)).max(),
         np.abs(A_bar - fd_grad(lambda Av: loss(Av @ W), A)).max()))

# What this chapter adds: weight tying breaks the assumption that A is not a
# function of W, and the two contributions ADD.
V, d, T = 9, 5, 7
ids = rng.integers(0, V, T)
W_E = rng.standard_normal((V, d))
ctx = rng.standard_normal((T, d))          # stands in for everything between


def tied_loss(WE):
    h = WE[ids] + ctx                      # the embedding leg
    logits = h @ WE.T                      # the unembedding leg, W_U = W_E^T
    z = logits - logits.max(axis=1, keepdims=True)
    return float(-(z[np.arange(T), ids] - np.log(np.exp(z).sum(axis=1))).mean())


exact = fd_grad(tied_loss, W_E, h=1e-6)

# The two legs, written separately so the omission is visible.
h = W_E[ids] + ctx
logits = h @ W_E.T
p = np.exp(logits - logits.max(axis=1, keepdims=True))
p /= p.sum(axis=1, keepdims=True)
dlogits = p.copy()
dlogits[np.arange(T), ids] -= 1.0
dlogits /= T
unembed_leg = dlogits.T @ h                # d L / d W_U, transposed back
dh = dlogits @ W_E
embed_leg = np.zeros_like(W_E)
np.add.at(embed_leg, ids, dh)              # scatter-add from the lookup, D-2.1

two_term = unembed_leg + embed_leg
assert np.abs(two_term - exact).max() < 1e-7, np.abs(two_term - exact).max()
rel = (np.linalg.norm(unembed_leg - exact, "fro") / np.linalg.norm(exact, "fro"))
assert rel > 0.1, rel
print("tied embedding: two-term gradient matches to %.2e; the one-term version "
      "is wrong by %.0f%% relative Frobenius"
      % (np.abs(two_term - exact).max(), 100 * rel))
'''

S2 = FD + r'''
SEED = 7002
rng = np.random.default_rng(SEED)
s = 12

Z = rng.standard_normal((s, s)) * 2.0
mask = np.triu(np.ones((s, s), dtype=bool), 1)
Z_masked = np.where(mask, -np.inf, Z)


def softmax_rows(z):
    m = np.max(np.where(np.isfinite(z), z, -np.inf), axis=1, keepdims=True)
    e = np.exp(np.where(np.isfinite(z), z - m, -np.inf))
    return e / e.sum(axis=1, keepdims=True)


P = softmax_rows(Z_masked)
Pbar = rng.standard_normal((s, s))

# Equation (7.7): two elementwise products and one row reduction, O(s^2).
contracted = P * (Pbar - (P * Pbar).sum(axis=1, keepdims=True))

# The explicit route of step 6: build J = diag(p) - p p^T for EVERY row and
# contract.  Same answer, one factor of s more arithmetic and memory.
explicit = np.empty((s, s))
for i in range(s):
    p = P[i]
    J = np.diag(p) - np.outer(p, p)
    explicit[i] = J @ Pbar[i]              # J is symmetric, so no transpose
assert np.abs(contracted - explicit).max() < 1e-14, np.abs(contracted - explicit).max()
print("contracted O(s) form vs explicit O(s^2) Jacobian: max abs %.2e"
      % np.abs(contracted - explicit).max())

# and both against a central difference through the softmax itself.
def f(z):
    return float((Pbar * softmax_rows(np.where(mask, -np.inf, z))).sum())


num = np.zeros((s, s))
h = 1e-6
for i in range(s):
    for j in range(s):
        if mask[i, j]:
            continue
        Zp = Z.copy(); Zp[i, j] += h
        Zm = Z.copy(); Zm[i, j] -= h
        num[i, j] = (f(Zp) - f(Zm)) / (2 * h)
assert np.abs(contracted - num).max() < 1e-8, np.abs(contracted - num).max()

# Step 7: the softmax gradient sees only contrast.
shifted = P * ((Pbar + 4.2) - (P * (Pbar + 4.2)).sum(axis=1, keepdims=True))
assert np.abs(shifted - contracted).max() < 1e-13

# The assumption, and the failure mode.  With -inf the masked probability is
# exactly zero, so (7.7) multiplies by zero and no re-masking is needed.
assert (P[mask] == 0.0).all()
assert (contracted[mask] == 0.0).all()

# With a large negative number instead, the masked probability is merely small.
# The factor (g_i - p.g) beside it is O(1), so masked positions accumulate a
# non-zero gradient every step: a future-information leak.  Whether a given
# constant underflows to zero depends on the dtype and on the row's maximum,
# which is exactly why -inf is the rule rather than a preference.
for m in (-30.0, -60.0, -90.0):
    P_big = softmax_rows(np.where(mask, m, Z))
    leak = P_big * (Pbar - (P_big * Pbar).sum(axis=1, keepdims=True))
    assert (P_big[mask] > 0).all(), m
    assert (np.abs(leak[mask]) > 0).all(), m
    assert np.abs(leak[mask]).max() < 1e-6
    print("mask %.0f: masked p up to %.1e, masked gradient up to %.1e (not zero)"
          % (m, P_big[mask].max(), np.abs(leak[mask]).max()))
print("mask -inf: masked p %.1e, masked gradient %.1e"
      % (P[mask].max(), np.abs(contracted[mask]).max()))

# Step 6's arithmetic, at the chapter's s.  Both ratios are exactly s.
S = 8192
flops_contracted = 4 * S * S               # two products and one reduction
flops_explicit = 4 * S ** 3
mem_contracted = 4 * S                     # one length-s vector, fp32
mem_explicit = 4 * S * S
assert flops_explicit // flops_contracted == S == mem_explicit // mem_contracted
assert round(flops_explicit / 1e12, 1) == 2.2
assert round(flops_contracted / 1e6) == 268
assert round(mem_explicit / 1e6) == 268 and round(mem_contracted / 1e3, 1) == 32.8
print("at s = %d: %.1f TFLOP against %.0f MFLOP, %.0f MB against %.1f kB"
      % (S, flops_explicit / 1e12, flops_contracted / 1e6,
         mem_explicit / 1e6, mem_contracted / 1e3))
'''

S3 = FD + r'''
SEED = 7003
rng = np.random.default_rng(SEED)
s, d = 7, 10

X = rng.standard_normal((s, d)) * 1.6
gamma = 1.0 + 0.3 * rng.standard_normal(d)
Ybar = rng.standard_normal((s, d))


def rmsnorm(x, g):
    r = np.sqrt((x * x).mean(axis=-1, keepdims=True))
    return x / r * g


# Equation (7.8), per token: a projector, then a scaling by 1/r.
r = np.sqrt((X * X).mean(axis=1, keepdims=True))
Xhat = X / np.linalg.norm(X, axis=1, keepdims=True)
Xbar = np.empty_like(X)
for t in range(s):
    P = np.eye(d) - np.outer(Xhat[t], Xhat[t])
    Xbar[t] = P @ (gamma * Ybar[t]) / r[t, 0]
    assert np.abs(P @ P - P).max() < 1e-14         # idempotent
    assert np.abs(P @ X[t]).max() < 1e-12          # annihilates x itself

num = fd_grad(lambda Xv: float((Ybar * rmsnorm(Xv, gamma)).sum()), X)
assert np.abs(Xbar - num).max() < 1e-8, np.abs(Xbar - num).max()

# Equation (7.9): the gain gradient is a SUM over tokens, because gamma is
# shared.  A per-token version has the right shape and is wrong by a factor of s.
gbar = (Ybar * (X / r)).sum(axis=0)
gnum = fd_grad(lambda gv: float((Ybar * rmsnorm(X, gv)).sum()), gamma)
assert np.abs(gbar - gnum).max() < 1e-8, np.abs(gbar - gnum).max()
per_token = Ybar[0] * (X[0] / r[0])
assert np.linalg.norm(per_token - gbar) > 0.1 * np.linalg.norm(gbar)
print("RMSNorm backward: dX %.2e, dgamma %.2e against central differences"
      % (np.abs(Xbar - num).max(), np.abs(gbar - gnum).max()))

# The 1/r in front is a contraction once the stream norm exceeds sqrt(d), which
# is what §7.6 leans on and what pre-norm produces by accident.
big = X * 4.0
assert np.sqrt((big * big).mean(axis=1)).min() > 1.0
scale = 1.0 / np.sqrt((big * big).mean(axis=1))
assert scale.max() < 1.0
print("stream scaled by 4: every 1/r is now %.3f or less, so the block's "
      "Jacobian is contracted" % scale.max())
'''

S4 = FD + r'''
SEED = 7004
rng = np.random.default_rng(SEED)
s, d, h_q, n_kv, d_h = 6, 12, 4, 2, 3
g = h_q // n_kv                                   # query heads per kv head

N = rng.standard_normal((s, d))                   # the normalised stream
W_Q = rng.standard_normal((d, h_q * d_h)) / np.sqrt(d)
W_K = rng.standard_normal((d, n_kv * d_h)) / np.sqrt(d)
W_V = rng.standard_normal((d, n_kv * d_h)) / np.sqrt(d)
W_O = rng.standard_normal((h_q * d_h, d)) / np.sqrt(h_q * d_h)
Abar = rng.standard_normal((s, d))                # upstream, held fixed
mask = np.triu(np.ones((s, s), dtype=bool), 1)


def softmax_rows(z):
    m = np.max(np.where(np.isfinite(z), z, -np.inf), axis=1, keepdims=True)
    e = np.exp(np.where(np.isfinite(z), z - m, -np.inf))
    return e / e.sum(axis=1, keepdims=True)


def forward(N, W_Q, W_K, W_V, W_O, keep=None):
    Q = (N @ W_Q).reshape(s, h_q, d_h)
    K = (N @ W_K).reshape(s, n_kv, d_h)
    V = (N @ W_V).reshape(s, n_kv, d_h)
    O = np.empty((s, h_q, d_h))
    Ps = []
    for i in range(h_q):
        j = i // g                                # the kv head this query uses
        S = Q[:, i] @ K[:, j].T / np.sqrt(d_h)
        P = softmax_rows(np.where(mask, -np.inf, S))
        Ps.append(P)
        O[:, i] = P @ V[:, j]
    Ocat = O.reshape(s, h_q * d_h)
    if keep is not None:
        keep.update(Q=Q, K=K, V=V, Ps=Ps, Ocat=Ocat)
    return Ocat @ W_O


cache = {}
out = forward(N, W_Q, W_K, W_V, W_O, cache)
loss = lambda o: float((Abar * o).sum())

# D-7.3, in order.  Output projection first (equation 7.20), then per head.
Wo_bar = cache["Ocat"].T @ Abar
Ocat_bar = Abar @ W_O.T
Obar = Ocat_bar.reshape(s, h_q, d_h)

Qbar = np.zeros((s, h_q, d_h))
Kbar = np.zeros((s, n_kv, d_h))                   # summed over the group
Vbar = np.zeros((s, n_kv, d_h))
Sbars = []
for i in range(h_q):
    j = i // g
    P = cache["Ps"][i]
    Vbar[:, j] += P.T @ Obar[:, i]                # (7.15), a SUM not a mean
    Pbar = Obar[:, i] @ cache["V"][:, j].T        # (7.15)
    Sbar = P * (Pbar - (P * Pbar).sum(axis=1, keepdims=True))   # (7.16)
    Sbars.append(Sbar)
    Qbar[:, i] = Sbar @ cache["K"][:, j] / np.sqrt(d_h)          # (7.17)
    Kbar[:, j] += Sbar.T @ cache["Q"][:, i] / np.sqrt(d_h)       # (7.17)

Wq_bar = N.T @ Qbar.reshape(s, h_q * d_h)
Wk_bar = N.T @ Kbar.reshape(s, n_kv * d_h)
Wv_bar = N.T @ Vbar.reshape(s, n_kv * d_h)
Nbar = (Qbar.reshape(s, -1) @ W_Q.T + Kbar.reshape(s, -1) @ W_K.T
        + Vbar.reshape(s, -1) @ W_V.T)            # (7.19), three reads, three terms

checks = {
    "dW_Q": (Wq_bar, fd_grad(lambda W: loss(forward(N, W, W_K, W_V, W_O)), W_Q)),
    "dW_K": (Wk_bar, fd_grad(lambda W: loss(forward(N, W_Q, W, W_V, W_O)), W_K)),
    "dW_V": (Wv_bar, fd_grad(lambda W: loss(forward(N, W_Q, W_K, W, W_O)), W_V)),
    "dW_O": (Wo_bar, fd_grad(lambda W: loss(forward(N, W_Q, W_K, W_V, W)), W_O)),
    "dN":   (Nbar,   fd_grad(lambda X: loss(forward(X, W_Q, W_K, W_V, W_O)), N)),
}
for name, (ana, num) in checks.items():
    e = np.abs(ana - num).max() / max(1.0, np.abs(num).max())
    assert e < 1e-7, (name, e)
    print("%-5s vs central difference: relative max abs %.2e" % (name, e))

# Step 4: masked entries carry exactly zero, with no re-masking in the backward.
for Sbar in Sbars:
    assert (Sbar[mask] == 0.0).all()

# Step 6: the transpose in dK is not decoration.  Causal masking makes S
# non-symmetric, so swapping the two expressions passes every shape check and
# is wrong.
assert np.abs(Sbars[0] - Sbars[0].T).max() > 1e-6
swapped = np.zeros_like(Kbar)
for i in range(h_q):
    swapped[:, i // g] += Sbars[i] @ cache["Q"][:, i] / np.sqrt(d_h)
assert swapped.shape == Kbar.shape
rel = np.linalg.norm(swapped - Kbar) / np.linalg.norm(Kbar)
assert rel > 0.5, rel
print("dropping the transpose in dK: same shape, %.0f%% wrong" % (100 * rel))

# Step 9: a broadcast forward is a sum backward.  The mean is the commonest bug
# in a hand-written grouped-query kernel, and it survives every shape assertion.
mean_Kbar, mean_Vbar = Kbar / g, Vbar / g
rel_k = np.linalg.norm((N.T @ mean_Kbar.reshape(s, -1)) - Wk_bar) / np.linalg.norm(Wk_bar)
rel_v = np.linalg.norm((N.T @ mean_Vbar.reshape(s, -1)) - Wv_bar) / np.linalg.norm(Wv_bar)
assert abs(rel_k - (1 - 1 / g)) < 1e-12 and abs(rel_v - (1 - 1 / g)) < 1e-12
assert rel_k > 0.4

# the other spelling of the same bug: assign rather than accumulate, so only
# the last query head in each group contributes at all.
last_only = np.zeros_like(Kbar)
for i in range(h_q):
    last_only[:, i // g] = Sbars[i].T @ cache["Q"][:, i] / np.sqrt(d_h)
rel_last = (np.linalg.norm((N.T @ last_only.reshape(s, -1)) - Wk_bar)
            / np.linalg.norm(Wk_bar))
assert rel_last > 0.4, rel_last
print("at g = %d, averaging makes dW_K and dW_V wrong by %.0f%% and %.0f%%; "
      "assigning instead of accumulating, by %.0f%%"
      % (g, 100 * rel_k, 100 * rel_v, 100 * rel_last))
'''

S5 = FD + r'''
SEED = 7005
rng = np.random.default_rng(SEED)
s, d, d_ff = 6, 10, 14

Nin = rng.standard_normal((s, d))
W_gate = rng.standard_normal((d, d_ff)) / np.sqrt(d)
W_up = rng.standard_normal((d, d_ff)) / np.sqrt(d)
W_down = rng.standard_normal((d_ff, d)) / np.sqrt(d_ff)
fbar = rng.standard_normal((s, d))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def silu(z):
    return z * sigmoid(z)


def dsilu(z):                                   # equation (7.11)
    sg = sigmoid(z)
    return sg * (1 + z * (1 - sg))


def forward(Nin, W_gate, W_up, W_down):
    G = Nin @ W_gate
    U = Nin @ W_up
    return (silu(G) * U) @ W_down


loss = lambda f: float((fbar * f).sum())

G = Nin @ W_gate
U = Nin @ W_up
A = silu(G) * U

Wd_bar = A.T @ fbar                             # (7.12)
Abar = fbar @ W_down.T
Ubar = Abar * silu(G)                           # (7.13)
Gbar = Abar * U * dsilu(G)                      # (7.13)
Wu_bar = Nin.T @ Ubar                           # (7.14)
Wg_bar = Nin.T @ Gbar
Nbar = Ubar @ W_up.T + Gbar @ W_gate.T          # (7.14), the two-path sum

for name, ana, num in (
        ("dW_down", Wd_bar, fd_grad(lambda W: loss(forward(Nin, W_gate, W_up, W)), W_down)),
        ("dW_up", Wu_bar, fd_grad(lambda W: loss(forward(Nin, W_gate, W, W_down)), W_up)),
        ("dW_gate", Wg_bar, fd_grad(lambda W: loss(forward(Nin, W, W_up, W_down)), W_gate)),
        ("dn", Nbar, fd_grad(lambda X: loss(forward(X, W_gate, W_up, W_down)), Nin))):
    e = np.abs(ana - num).max() / max(1.0, np.abs(num).max())
    assert e < 1e-7, (name, e)
    print("%-8s vs central difference: relative max abs %.2e" % (name, e))

# The two branches are NOT symmetric.  Writing the same expression twice is a
# copy-paste bug the shapes cannot catch, so it gets its own assertion.
assert Ubar.shape == Gbar.shape
assert np.linalg.norm(Ubar - Gbar) / np.linalg.norm(Ubar) > 0.5

# Equation (7.11) itself, and the two facts about it the chapter states.
zs = np.linspace(-8, 8, 200001)
h = 1e-6
assert np.abs(dsilu(zs) - (silu(zs + h) - silu(zs - h)) / (2 * h)).max() < 1e-8
assert np.abs(dsilu(zs) + dsilu(-zs) - 1.0).max() < 1e-14      # exact symmetry
assert abs(dsilu(0.0) - 0.5) < 1e-15
from scipy.optimize import brentq, minimize_scalar
root = brentq(dsilu, -4.0, -0.5, xtol=1e-13)
assert abs(root - (-1.2785)) < 1e-3, root
m = minimize_scalar(dsilu, bounds=(-6, -1), method="bounded",
                    options={"xatol": 1e-12})
assert abs(m.x - (-2.3994)) < 1e-3 and abs(m.fun - (-0.0998)) < 1e-4
assert abs(dsilu(-m.x) - 1.0998) < 1e-4                        # 1 - (-0.0998)
print("SiLU' is negative below %.4f, minimum %.4f at %.4f, and SiLU'(g) + "
      "SiLU'(-g) = 1 exactly" % (root, m.fun, m.x))
'''

S6 = r'''
import os

SEED = 7006
rng = np.random.default_rng(SEED)
d, L = 24, 32

# D-7.4 step 2, as a Monte-Carlo test of the bound rather than a re-derivation.
for delta in (0.02, 1 / L, 0.1):
    worst = 0.0
    for _ in range(200):
        prod = np.eye(d)
        for _ in range(L):
            J = rng.standard_normal((d, d))
            J *= delta / np.linalg.svd(J, compute_uv=False)[0]   # ||J||_2 = delta
            assert abs(np.linalg.svd(J, compute_uv=False)[0] - delta) < 1e-12
            prod = prod @ (np.eye(d) + J)
        worst = max(worst, np.linalg.svd(prod - np.eye(d), compute_uv=False)[0])
    bound = (1 + delta) ** L - 1
    assert worst <= bound, (delta, worst, bound)
    print("delta %.4f: measured drift %.4f, bound %.4f, slack %.1fx"
          % (delta, worst, bound, bound / worst))

# Step 3, read off the bound.  O(1) exactly when delta = O(1/L).
assert round((1 + 1 / L) ** L - 1, 3) == 1.677
assert round(np.e - 1, 3) == 1.718
assert (1 + 1 / L) ** L - 1 < np.e - 1
for Lx in (32, 1000, 100000):
    assert (1 + 1 / Lx) ** Lx - 1 < np.e - 1        # bounded at any depth
assert round((1 + 0.1) ** 32 - 1, 1) == 20.1        # and exponential otherwise
print("at delta = 1/L the bound is %.3f for L = %d and rises only to e - 1 = "
      "%.3f; at delta = 0.1 it is %.1f" % ((1 + 1 / L) ** L - 1, L, np.e - 1,
                                           (1 + 0.1) ** 32 - 1))

# The measurement F-7.3 plots, from the committed file rather than retyped.
def repo_file(*parts):
    """Works whether the notebook is run from notebooks/ or from the root."""
    for base in ("..", "."):
        p = os.path.join(base, *parts)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(os.path.join(*parts))


D = np.load(repo_file("figs", "data", "fig73.npz"))
assert int(D["L"]) == L
for key, ratio in (("pre_plain", 12.5), ("post_plain", 45.7),
                   ("pre_scaled", 1.33), ("post_scaled", 1.16)):
    y = D[key]
    assert y.shape == (L + 1,) and abs(y[-1] - 1.0) < 1e-12   # normalised at the top
    r = y[0] / y[-1]
    assert abs(round(r, 1 if r > 2 else 2) - ratio) < 1e-9, (key, r)

implied = {k: (D[k][0] / D[k][-1]) ** (1 / L) - 1
           for k in ("pre_plain", "post_plain", "pre_scaled", "post_scaled")}
assert round(implied["pre_plain"], 3) == 0.082
assert round(implied["post_plain"], 3) == 0.127
assert implied["pre_scaled"] < 0.01 and implied["post_scaled"] < 0.01
# the direction is amplification, not decay, which is not the usual telling
assert D["pre_plain"][0] > 1 and D["post_plain"][0] > 1
assert D["post_plain"][0] / D["pre_plain"][0] > 3
# and the initialisation scaling matters more than where the norm sits
assert D["post_scaled"][0] < D["pre_plain"][0]
for k, v in implied.items():
    print("%-11s ratio %7.3f, implied delta %.4f%s"
          % (k, D[k][0] / D[k][-1], v, "  (delta << 1/L)" if v < 1 / L else ""))
'''

SECTIONS = [
    ("1", "The linear layer under tokens-as-rows",
     "With tokens as rows the two backward products are forced by the shapes, "
     "which is why the rule is worth recognising rather than memorising. What "
     "this chapter adds is the case Chapter 1 excludes: when the embedding is "
     "tied, the same matrix appears on two legs of the graph and the two "
     "contributions add. The cell checks both against central differences and "
     "measures how wrong the one-term version is.",
     S1),
    ("2", "Softmax backward without forming the Jacobian",
     "Equation (7.7) is two elementwise products and a row reduction, and it is "
     "the same map as contracting the explicit Jacobian row by row. The cell "
     "asserts that the O(s) contracted form and the O(s squared) explicit form "
     "agree to machine precision, checks both against a central difference, and "
     "then does step 6's arithmetic at the chapter's sequence length. The last "
     "block is the failure mode: masking with a large negative number instead "
     "of minus infinity leaves a non-zero gradient on masked positions.",
     S2),
    ("3", "RMSNorm, the third atom",
     "The remaining atom is Chapter 5's Jacobian, restated here in this "
     "chapter's notation because the block below uses it twice. Two properties "
     "are used repeatedly: the projector is idempotent and annihilates the "
     "input direction, and the gain gradient is a sum over tokens rather than a "
     "per-token quantity, since the gain is shared.",
     S3),
    ("4", "The attention backward pass",
     "One head at a time, then the output projection, then the grouped-query "
     "reduction, with nothing deferred. Every gradient is checked against a "
     "central difference on the full forward pass. Three things the shapes "
     "cannot catch get their own assertions: masked entries must be exactly "
     "zero, the transpose in the key gradient is load-bearing because causal "
     "masking makes the score matrix non-symmetric, and a broadcast forward is "
     "a sum backward and not a mean.",
     S4),
    ("5", "The feed-forward backward",
     "The gated FFN is the only place in the block where one upstream gradient "
     "fans out into two, and the two branches are not symmetric: one carries "
     "the activation and the other carries the other projection times the "
     "activation's derivative. The cell checks all four gradients against "
     "central differences, asserts that the two branch expressions differ (a "
     "strange-looking test until you have made the mistake), and verifies the "
     "SiLU derivative's exact symmetry and its negative region.",
     S5),
    ("6", "How far the product can drift from the identity",
     "The residual stack is a product of identity-plus-Jacobian factors, and "
     "the whole of D-7.4 is the observation that the drift from the identity is "
     "bounded by one plus delta to the L, minus one. The cell tests that bound "
     "by Monte Carlo at three values of delta, reads off the condition that "
     "delta be order one over L, and then loads the committed gradient-flow "
     "measurement behind Figure 7.3 and recovers the per-layer delta each "
     "configuration implies.",
     S6),
]
