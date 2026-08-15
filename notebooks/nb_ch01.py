"""Chapter 1 — The Toolkit in One Chapter.

Six sections, one per derivation the chapter carries a `\\repo` margin note for,
plus the two that set up the notation.  §3 to §6 are the contract: D-1.1, D-1.2,
D-1.3 and D-1.4 in that order.  Nothing here is transcribed from the page.  The
parameter ledger of §2 is read out of `arith/model_d.py`, so a second edition
that edits Table 1.2 either keeps this notebook green or is caught by it.
"""
from __future__ import annotations

CHAPTER = 1
SLUG = "toolkit"
TITLE = "The Toolkit in One Chapter"
BLURB = ("The conventions and the four identities that the rest of the book "
         "assumes without restating them: norms and the four subspaces, the "
         "Model D ledger, the softmax Jacobian, the p minus y collapse, Gibbs' "
         "inequality, and denominator layout.")

MD1 = """\
An inner product carries three quantities at once, the two lengths and the angle
between them, and every claim in §1.1 and §1.2 is a statement about one of the
three. The cell below builds one small weight matrix and reads the four
subspaces off its SVD, then checks that the Frobenius and spectral norms measure
what §1.2 says they measure (the sum of squared singular values, and the largest
stretch the map applies to any input). The last block is the rank-nullity
statement in the form Chapter 13 uses it: a LoRA update of rank r is blind to
d_in minus r directions, which is arithmetic and not an opinion."""

CODE1 = """\
from arith.model_d import MODEL_D

rng = np.random.default_rng(1)
d_in, d_out = 6, 4
W = rng.normal(size=(d_in, d_out))

U, s, Vt = np.linalg.svd(W)                       # (1.5): W = U S V^T
assert np.allclose(U[:, :len(s)] @ np.diag(s) @ Vt, W, atol=1e-12)

# (1.11) and (1.12): the two matrix norms, both read off the same spectrum.
assert abs(np.linalg.norm(W, "fro") - np.sqrt((s ** 2).sum())) < 1e-12
assert abs(np.linalg.norm(W, 2) - s[0]) < 1e-12

# The spectral norm as a maximum stretch, in the row convention x -> xW.
x = rng.normal(size=(20000, d_in))
stretch = np.linalg.norm(x @ W, axis=1) / np.linalg.norm(x, axis=1)
assert stretch.max() <= s[0] + 1e-12
top = U[:, 0]                                     # the input direction that attains it
assert abs(np.linalg.norm(top @ W) / np.linalg.norm(top) - s[0]) < 1e-12

# (1.6): rank + nullity = d_in, on a deliberately rank-deficient map.
Wr = rng.normal(size=(d_in, 3)) @ rng.normal(size=(3, d_out))
sr = np.linalg.svd(Wr, compute_uv=False)
rank = int((sr > 1e-10).sum())
nullity = d_in - rank
assert rank == 3 and rank + nullity == d_in
Ur = np.linalg.svd(Wr)[0]
null_dir = Ur[:, rank:] @ rng.normal(size=nullity)
assert np.linalg.norm(null_dir @ Wr) < 1e-10, "the null space is what the map discards"

# (1.7) and (1.8): Cauchy-Schwarz, tight exactly for parallel vectors.
a = rng.normal(size=(5000, 8))
b = rng.normal(size=(5000, 8))
ip = np.einsum("ij,ij->i", a, b)
assert (np.abs(ip) <= np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12).all()
u = rng.normal(size=8)
assert abs(abs(np.dot(u, 2.5 * u)) - np.linalg.norm(u) * np.linalg.norm(2.5 * u)) < 1e-12

# (1.9): the projector is symmetric, idempotent, and kills nothing it keeps.
P = np.outer(u, u) / np.dot(u, u)
assert np.allclose(P, P.T, atol=1e-14)
assert np.allclose(P @ P, P, atol=1e-14)
assert np.allclose(P @ u, u, atol=1e-12)
v = rng.normal(size=8)
assert abs(np.dot(P @ v, v - P @ v)) < 1e-12                       # orthogonal split
assert abs(np.linalg.norm(v) ** 2
           - np.linalg.norm(P @ v) ** 2 - np.linalg.norm(v - P @ v) ** 2) < 1e-12

# Chapter 13's constraint, stated with Chapter 1's theorem, at Model D's width.
r = 16
B = rng.normal(size=(MODEL_D.d, r))
A = rng.normal(size=(r, MODEL_D.d))
dW = B @ A
assert np.linalg.matrix_rank(dW) == r
blind = MODEL_D.d - r                             # directions the update cannot reach
assert blind == 4080 and blind / MODEL_D.d > 0.99

# And §1.1's reading of the column space: one head writes into at most d_h of d.
assert MODEL_D.d == MODEL_D.h * MODEL_D.d_h
assert abs(MODEL_D.d_h / MODEL_D.d - 1 / 32) < 1e-15
print(f"singular values {s}")
print(f"one head writes into {MODEL_D.d_h} of {MODEL_D.d} directions"
      f" = 1/{MODEL_D.d // MODEL_D.d_h} of the stream")"""

MD2 = """\
The arithmetic box at the end of the chapter counts Model D end to end, and the
count is worth reproducing because every later chapter's memory and FLOP figure
starts from it. Each line below recomputes a row of that table from the
hyperparameters in `arith/model_d.py` and asserts it against the function the
book cites, so the table and this notebook cannot disagree. The last block is
the grouped-query reading: rebuilding the same config with n_kv equal to h
recovers the multi-head counterfactual, and the difference is the saving the box
quotes before Chapter 11 has said a word about the KV cache."""

CODE2 = """\
from dataclasses import replace
from arith.model_d import (MODEL_D, attention_params, per_layer, non_embedding,
                           embedding, total_params)

c = MODEL_D
ap = attention_params(c)

# Row by row, against the formulas printed in the box.
assert ap["W_Q"] == c.d * c.h * c.d_h
assert ap["W_K"] == ap["W_V"] == c.d * c.n_kv * c.d_h
assert ap["W_O"] == c.h * c.d_h * c.d
attn = sum(ap.values())

pl = per_layer(c)
assert pl["attention"] == attn
assert pl["mlp"] == 3 * c.d * c.d_ff                      # SwiGLU: gate, up, down
assert pl["norms"] == 2 * c.d                             # two RMSNorm gain vectors
assert pl["total"] == attn + 3 * c.d * c.d_ff + 2 * c.d

assert non_embedding(c) == pl["total"] * c.L + c.d        # + the final norm
assert embedding(c) == c.V * c.d
assert total_params(c) == non_embedding(c) + 2 * embedding(c)

# The two readings the box asks for.
ffn_share = pl["mlp"] / pl["total"]
assert abs(ffn_share - 0.8077) < 5e-4 and round(100 * ffn_share) == 81
emb_share = 2 * embedding(c) / total_params(c)
assert abs(100 * emb_share - 13.08) < 5e-3

# Grouped-query attention, as a counterfactual rather than a quoted number.
mha = replace(c, n_kv=c.h)
attn_mha = sum(attention_params(mha).values())
assert attn_mha == 4 * c.d * c.h * c.d_h
assert attn == 2 * c.d * c.d_h * (c.h + c.n_kv)
saved = attn_mha - attn
assert saved == 2 * c.d * c.d_h * (c.h - c.n_kv)
assert abs(saved * c.L / 1e9 - 0.81) < 5e-3

# Bytes are count times precision, which is the whole of the fitting question.
gb_bf16 = 2 * total_params(c) / 1e9
assert abs(gb_bf16 - 16.06) < 5e-3 and gb_bf16 > 16.0, "will not fit a 16 GB card"

# Tying the two embedding matrices is the only lever on that number in Chapter 2.
tied_total = non_embedding(c) + embedding(c)
assert total_params(c) - tied_total == embedding(c)
assert abs(100 * embedding(c) / total_params(c) - 6.54) < 5e-3

print(f"per layer      {pl['total']:,}")
print(f"non-embedding  {non_embedding(c):,}")
print(f"total          {total_params(c):,}   ({gb_bf16:.2f} GB in bf16)")
print(f"FFN share of a layer {100*ffn_share:.1f}% ; GQA saves {saved*c.L/1e9:.2f} B")"""

MD3 = """\
D-1.1 makes three claims about softmax: the image is the open simplex, the map
is invariant to adding a constant to every logit, and the Jacobian is
diag(p) minus p p transpose. The first two are checked directly and the third is
checked twice, once against a central difference and once against its own stated
properties (it annihilates the all-ones vector, it is symmetric, and it is
positive semidefinite with rank one less than the number of logits). The last
block is step 8 and the failure mode beneath it, at Model D's vocabulary size:
subtracting the maximum is not a refinement, it is the difference between a
probability vector and a page of NaNs."""

CODE3 = """\
from arith.model_d import MODEL_D

def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

rng = np.random.default_rng(3)
n = 7
z = rng.normal(scale=2.0, size=n)
p = softmax(z)

# Steps 1 and 2: strictly interior to the simplex, never a vertex.
assert (p > 0).all() and abs(p.sum() - 1.0) < 1e-15
assert p.max() < 1.0

# Step 3: shift invariance, so the map is not injective.
for c_shift in (-50.0, -1.0, 12.5, 300.0):
    assert np.allclose(softmax(z + c_shift), p, atol=1e-15)

# Steps 4 to 7: the Jacobian, against a central difference.
J = np.diag(p) - np.outer(p, p)
num = np.zeros((n, n))
eps = 1e-6
for j in range(n):
    e = np.zeros(n)
    e[j] = eps
    num[:, j] = (softmax(z + e) - softmax(z - e)) / (2 * eps)
assert np.abs(J - num).max() < 1e-9, np.abs(J - num).max()

# Step 7's check: the singular direction is the all-ones vector, and only it.
assert np.abs(J @ np.ones(n)).max() < 1e-15
assert np.allclose(J, J.T, atol=1e-16)
lam = np.linalg.eigvalsh(J)
assert lam.min() > -1e-15 and abs(lam.min()) < 1e-15      # PSD, one exact zero
assert (lam[1:] > 1e-9).all() and np.linalg.matrix_rank(J, tol=1e-9) == n - 1

# Temperature (1.16): every pairwise log-odds ratio scales by 1/T.
for T in (0.25, 1.0, 4.0):
    pT = softmax(z / T)
    assert abs(np.log(pT[0] / pT[1]) - (z[0] - z[1]) / T) < 1e-12
zs = np.sort(z)[::-1]
T_cold = (zs[0] - zs[1]) / 8.0
cold = softmax(z / T_cold)
assert cold.argmax() == z.argmax()
assert 0.0 < 1.0 - cold.max() < 1e-2, "concentrating, and step 1 says never arriving"
assert softmax(z / 1e-4).max() == 1.0, "though in floating point it does arrive"

# Step 8, at V = 128256, and the failure mode directly beneath it.
V = MODEL_D.V
big = rng.normal(scale=3.0, size=V) + 900.0
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(big) / np.exp(big).sum()
assert not np.isfinite(naive).all(), "the unshifted form overflows at these logits"
stable = softmax(big)
assert np.isfinite(stable).all() and abs(stable.sum() - 1.0) < 1e-12 and (stable > 0).all()

# A fully masked row: -inf everywhere makes the shift itself -inf, and 0/0 is NaN.
with np.errstate(invalid="ignore"):
    masked = softmax(np.full(8, -np.inf))
assert np.isnan(masked).all(), "this is the NaN that arrives 200 steps in, on one batch"
print(f"max |J - central difference| = {np.abs(J - num).max():.2e}")
print(f"V = {V:,}: stable sum {stable.sum():.12f}, naive finite? {np.isfinite(naive).all()}")"""

MD4 = """\
The collapse of softmax cross-entropy to a subtraction is the identity the book
writes without comment from Chapter 7 onwards, so it is worth checking by both
of the routes D-1.2 gives. The direct route differentiates the log-sum-exp form,
and the indirect route pushes the gradient of the loss with respect to p back
through the Jacobian of §3. They agree with each other and with a central
difference, and the result has the shape of z, which is §1.5's convention doing
its job. The final block is the failure mode: computed as two separate layers
the loss overflows on a confident wrong prediction, while the fused form is
finite and exact."""

CODE4 = """\
rng = np.random.default_rng(4)
V_small = 9
z = rng.normal(scale=2.0, size=V_small)
target = 3
y = np.zeros(V_small)
y[target] = 1.0

def logsumexp(z):
    m = z.max()
    return m + np.log(np.exp(z - m).sum())

def loss(z):
    return -z[target] + logsumexp(z)                       # step 1, the collapsed form

p = softmax(z)
assert abs(loss(z) + np.log(p[target])) < 1e-12            # same as -log p_y

g = p - y                                                  # step 4
num = np.zeros(V_small)
eps = 1e-6
for j in range(V_small):
    e = np.zeros(V_small)
    e[j] = eps
    num[j] = (loss(z + e) - loss(z - e)) / (2 * eps)
assert np.abs(g - num).max() < 1e-9, np.abs(g - num).max()

# Step 6: the same answer through the Jacobian of D-1.1, which is the point of
# doing it twice.  dL/dp = -y/p, and J is symmetric so J^T = J.
J = np.diag(p) - np.outer(p, p)
via_jacobian = J.T @ (-y / p)
assert np.abs(via_jacobian - g).max() < 1e-12

# Step 5: denominator layout.  The gradient has the shape of the logits, so the
# assertion Chapter 1 promises CI would run is a one-liner.
assert g.shape == z.shape

# It sums to zero, which is step 3 of D-1.1 again: no gradient along all-ones.
assert abs(g.sum()) < 1e-15

# The batched version, mean over tokens, which is what a training step uses.
s = 16
Z = rng.normal(scale=2.0, size=(s, V_small))
ids = rng.integers(0, V_small, size=s)
P = softmax(Z)
Y = np.eye(V_small)[ids]
G = (P - Y) / s
L = float(np.mean([-Z[t, ids[t]] + logsumexp(Z[t]) for t in range(s)]))
numG = np.zeros_like(Z)
for t in range(s):
    for j in range(V_small):
        E = np.zeros_like(Z)
        E[t, j] = eps
        f = lambda M: np.mean([-M[u, ids[u]] + logsumexp(M[u]) for u in range(s)])
        numG[t, j] = (f(Z + E) - f(Z - E)) / (2 * eps)
assert np.abs(G - numG).max() < 1e-9
assert G.shape == Z.shape and abs(L - float(-np.log(P[np.arange(s), ids]).mean())) < 1e-12

# The failure mode: softmax and log as separate layers, on a confident mistake.
z_conf = np.array([0.0, 900.0, -900.0])
with np.errstate(divide="ignore"):
    separate = -np.log(softmax(z_conf)[0])
fused = -z_conf[0] + logsumexp(z_conf)
assert not np.isfinite(separate) and np.isfinite(fused)
assert abs(fused - 900.0) < 1e-9
print(f"max |analytic - central difference| = {np.abs(g - num).max():.2e}")
print(f"separate layers: {separate}   fused: {fused:.4f}")"""

MD5 = """\
Gibbs' inequality is the reason a training loss has a floor, and the
decomposition beneath it is the reason that floor is the entropy of the data
rather than zero. Both are checked here on random pairs rather than on one
example, because an inequality that holds for one pair is not evidence. The
equality case is checked in the direction that matters, that a zero divergence
forces the two distributions to coincide, and the asymmetry is made concrete
because §15.3 turns on which of the two orderings a method optimises."""

CODE5 = """\
from scipy.special import rel_entr, entr
from arith.model_d import MODEL_D

rng = np.random.default_rng(5)

def H(p):
    return float(-(p[p > 0] * np.log(p[p > 0])).sum())

def CE(p, q):
    return float(-(p[p > 0] * np.log(q[p > 0])).sum())

def KL(p, q):
    return float((p[p > 0] * np.log(p[p > 0] / q[p > 0])).sum())

# Gibbs, over a thousand random pairs at assorted sizes and concentrations.
worst = 0.0
for _ in range(1000):
    k = int(rng.integers(2, 40))
    p = rng.dirichlet(np.full(k, rng.uniform(0.2, 5.0)))
    q = rng.dirichlet(np.full(k, rng.uniform(0.2, 5.0)))
    kl = KL(p, q)
    assert kl > -1e-12, "Gibbs' inequality"
    worst = min(worst, kl)
    # The decomposition H(p,q) = H(p) + KL(p||q), which is steps 6 to 8.
    assert abs(CE(p, q) - H(p) - kl) < 1e-12
    # And the SciPy reference, independently implemented.
    assert abs(kl - float(rel_entr(p, q).sum())) < 1e-12
    assert abs(H(p) - float(entr(p).sum())) < 1e-12
assert worst > -1e-12

# Equality if and only if p = q, in both directions.
p = rng.dirichlet(np.ones(12))
assert abs(KL(p, p)) < 1e-15 and abs(CE(p, p) - H(p)) < 1e-15
q = p.copy()
q[0] += 1e-3
q[1] -= 1e-3
assert KL(p, q) > 1e-7, "a perturbation of 1e-3 must produce a strictly positive KL"

# KL is not a distance: not symmetric, and no triangle inequality.
a = np.array([0.90, 0.09, 0.01])
b = np.array([0.05, 0.15, 0.80])
assert abs(KL(a, b) - KL(b, a)) > 0.5
assert KL(a, b) > 0 and KL(b, a) > 0

# At Model D's vocabulary: the uniform distribution is the maximum-entropy point,
# and the divergence from it is exactly the entropy shortfall.
V = MODEL_D.V
u = np.full(V, 1.0 / V)
assert abs(H(u) - np.log(V)) < 1e-9
model = rng.dirichlet(np.full(V, 0.5))
assert H(model) < H(u)
assert abs(KL(model, u) - (np.log(V) - H(model))) < 1e-9
assert abs(CE(model, u) - np.log(V)) < 1e-9, "coding by a uniform code costs log V, always"
print(f"H(uniform over V={V:,}) = {np.log(V):.4f} nats = {np.log2(V):.4f} bits")
print(f"KL(a||b) = {KL(a, b):.4f} but KL(b||a) = {KL(b, a):.4f}")"""

MD6 = """\
Denominator layout is a contract rather than a taste, and the reason is that the
gradient descent step has to be a legal subtraction. This section derives both
halves of D-1.4 numerically, against a central difference on a scalar loss, and
then audits the shapes. The final block is the failure mode the chapter warns
about: under numerator layout the same expression is right up to a transpose,
which raises an error on a rectangular weight and runs silently on a square one,
which is why it survives to production."""

CODE6 = """\
rng = np.random.default_rng(6)
s, d_in, d_out = 5, 4, 3
X = rng.normal(size=(s, d_in))
W = rng.normal(size=(d_in, d_out))
b = rng.normal(size=d_out)
M = rng.normal(size=(s, d_out))                # an arbitrary linear read-out of Y

def forward(X, W, b):
    Y = X @ W + b
    return float((M * Y).sum()), Y             # so dL/dY = M exactly

L, Y = forward(X, W, b)
dY = M

dW = X.T @ dY                                  # step 5
dX = dY @ W.T                                  # step 6
db = np.ones(s) @ dY                           # the bias case in the assumptions

def central(f, A, eps=1e-6):
    g = np.zeros_like(A)
    it = np.nditer(A, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        A[i] += eps
        hi = f()
        A[i] -= 2 * eps
        lo = f()
        A[i] += eps
        g[i] = (hi - lo) / (2 * eps)
        it.iternext()
    return g

assert np.abs(dW - central(lambda: forward(X, W, b)[0], W)).max() < 1e-8
assert np.abs(dX - central(lambda: forward(X, W, b)[0], X)).max() < 1e-8
assert np.abs(db - central(lambda: forward(X, W, b)[0], b)).max() < 1e-8

# Step 7, the shape audit, and (1.19), the reason it has to come out this way.
assert dW.shape == W.shape and dX.shape == X.shape and db.shape == b.shape
eta = 0.01
W_new = W - eta * dW                           # a legal subtraction, by construction
assert W_new.shape == W.shape
assert forward(X, W_new, b)[0] < L, "one step downhill, since dL/dW points uphill"

# The failure mode.  Numerator layout transposes both results.
try:
    _ = W - eta * dW.T
    raised = False
except ValueError:
    raised = True
assert raised, "on a rectangular weight the mistake is caught by the shapes"

# On a square weight it is not caught by anything, and the answer is simply wrong.
Wsq = rng.normal(size=(4, 4))
Msq = rng.normal(size=(s, 4))
Lsq = lambda Wq: float((Msq * (X @ Wq)).sum())
dWsq = X.T @ Msq
assert np.abs(dWsq - central(lambda: Lsq(Wsq), Wsq)).max() < 1e-8
assert (Wsq - eta * dWsq.T).shape == Wsq.shape          # type-checks, and is wrong
assert np.abs(dWsq - dWsq.T).max() > 1e-3
assert Lsq(Wsq - eta * dWsq) < Lsq(Wsq) < Lsq(Wsq + eta * dWsq)
print(f"max |dW - central difference| = "
      f"{np.abs(dW - central(lambda: forward(X, W, b)[0], W)).max():.2e}")
print(f"loss {L:.6f} -> {forward(X, W_new, b)[0]:.6f} after one step of size {eta}")"""

SECTIONS = [
    ("1", "Norms, inner products, and the four subspaces", MD1, CODE1),
    ("2", "The Model D parameter ledger", MD2, CODE2),
    ("3", "The softmax map and its Jacobian", MD3, CODE3),
    ("4", "The gradient of softmax cross-entropy is p - y", MD4, CODE4),
    ("5", "Gibbs' inequality, and H(p,q) = H(p) + KL(p||q)", MD5, CODE5),
    ("6", "Denominator layout, and the gradient of a linear layer", MD6, CODE6),
]
