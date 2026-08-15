"""Chapter 3 — Attention from First Principles.

Four sections, matching the chapter's four `\\repo` margin notes exactly:
D-3.1 (three projections), D-3.2 (the scaling and the Jacobian bound), D-3.3
(multi-head as additive low-rank writes) and D-3.4 (Nadaraya-Watson).

§2 also reads `figs/data/ch11_scores.npz`, one trained layer's queries, keys and
scores, committed so the measurement reproduces without training anything.  The
i.i.d. hypotheses behind the scaling factor are false in that data, and the
notebook says so with numbers rather than adjectives.
"""
from __future__ import annotations

CHAPTER = 3
SLUG = "attention"
TITLE = "Attention from First Principles"
BLURB = ("Soft lookup on an unordered set, the constant in front of it, the "
         "sense in which h heads are h low-rank additive writes into one "
         "stream, and the 1964 estimator that attention turns out to be.")

MD1 = """\
Each of the three projections in attention is forced by a separate requirement,
and the way to see that is to remove one and watch what becomes unrepresentable.
The cell below builds the hard lookup, recovers it as the zero-temperature limit
of the soft one, and then checks the two structural claims: factoring the
bilinear form through a width of d_h caps its rank at d_h, and tying the query
and key projections makes the score matrix symmetric, so no directional relation
survives. The final block assembles equation (3.6) and checks it against an
explicit row-by-row weighted average."""

CODE1 = """\
def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

rng = np.random.default_rng(31)
s, d, d_h = 6, 12, 4
X = rng.normal(size=(s, d))
W_Q = rng.normal(size=(d, d_h)) / np.sqrt(d)
W_K = rng.normal(size=(d, d_h)) / np.sqrt(d)
W_V = rng.normal(size=(d, d_h)) / np.sqrt(d)

Q, K, V = X @ W_Q, X @ W_K, X @ W_V
Z = Q @ K.T / np.sqrt(d_h)

# Step 1 to 3: hard lookup is the zero-temperature limit of the soft one, and
# the soft one is differentiable everywhere while the hard one is not.
hard = V[Z.argmax(axis=1)]
for T in (1.0, 1e-2, 1e-4):
    soft = softmax(Z / T) @ V
    if T == 1e-4:
        assert np.abs(soft - hard).max() < 1e-8
    assert np.isfinite(soft).all()
A = softmax(Z)
assert np.allclose(A.sum(axis=1), 1.0, atol=1e-15) and (A > 0).all()

# Step 5: B = W_Q W_K^T costs 2 d d_h parameters instead of d^2, and its rank is
# capped at d_h.  Both halves are arithmetic.
B = W_Q @ W_K.T
assert B.shape == (d, d)
assert np.linalg.matrix_rank(B, tol=1e-10) == min(d_h, d) == d_h
assert 2 * d * d_h < d * d
assert np.abs(X @ B @ X.T - Q @ K.T).max() < 1e-12, "the factored score IS the bilinear one"

# Step 6: with W_Q = W_K the score matrix is symmetric before masking, so
# "it attends to cat" and "cat attends to it" cannot differ.  With two distinct
# projections it is not symmetric, and that asymmetry is the point.
Z_tied = (X @ W_Q) @ (X @ W_Q).T / np.sqrt(d_h)
assert np.abs(Z_tied - Z_tied.T).max() < 1e-12
assert np.abs(Z - Z.T).max() > 0.1
i, j = 1, 4
assert abs(softmax(Z)[i, j] - softmax(Z)[j, i]) > 1e-3

# Step 7: nothing forces the matched vector to be the summed vector.  Tying
# W_V to W_K makes the retrieved content equal the matching signature.
out_tied_v = A @ (X @ W_K)
assert np.abs(out_tied_v - A @ V).max() > 0.1

# Step 8: equation (3.6), against an explicit per-row weighted average.
attn = A @ V
manual = np.zeros_like(attn)
for r in range(s):
    w = np.exp(Z[r] - Z[r].max())
    w = w / w.sum()
    manual[r] = sum(w[j2] * V[j2] for j2 in range(s))
assert np.abs(attn - manual).max() < 1e-12

# Every output row is a convex combination of value rows, so it cannot leave
# their convex hull.  That is what "retrieval" means here.
assert (attn.max(axis=0) <= V.max(axis=0) + 1e-12).all()
assert (attn.min(axis=0) >= V.min(axis=0) - 1e-12).all()
print(f"rank(B) = {np.linalg.matrix_rank(B, tol=1e-10)} <= d_h = {d_h};"
      f" parameters {2*d*d_h} vs {d*d} for a full bilinear form")
print(f"asymmetry of the untied score matrix: max |Z - Z^T| = {np.abs(Z - Z.T).max():.3f}")"""

MD2 = """\
Part (a) of D-3.2 is a variance calculation under two independence hypotheses,
and part (b) says what a large logit gap does to the softmax Jacobian. Both are
checked here, the first by sampling at four head widths and the second against
the exact spectral norm over thousands of random logit vectors. The last block
is the part the chapter is careful about: on one trained layer, committed under
`figs/data/`, the coordinates of q are strongly correlated and the entries are
not unit variance, so the realised logit spread is more than twice what the
hypotheses predict. The scaling factor is still the right constant, and its
derivation is still false in the trained model."""

CODE2 = """\
def find_data(name):
    for base in (os.path.join(os.getcwd(), ".."), os.getcwd()):
        p = os.path.abspath(os.path.join(base, "figs", "data", name))
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)

rng = np.random.default_rng(32)

# Part (a), steps 1 to 5: Var<q,k> = d_h, so the scaled logit has unit variance.
for d_h in (8, 32, 128, 512):
    q = rng.standard_normal((40000, d_h))
    k = rng.standard_normal((40000, d_h))
    ip = np.einsum("ij,ij->i", q, k)
    assert abs(ip.mean()) < 4 * np.sqrt(d_h / 40000)
    assert abs(ip.var() / d_h - 1.0) < 0.05, (d_h, ip.var() / d_h)
    scaled = ip / np.sqrt(d_h)
    assert abs(scaled.var() - 1.0) < 0.05
    assert abs(scaled.std() - 1.0) < 0.03
print("unit logit variance at every head width tested")

# Step 6: the magnitude check the chapter makes at Model D's head width.
from arith.model_d import MODEL_D
assert abs(np.sqrt(MODEL_D.d_h) - 11.31) < 5e-3

def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

# Part (b), steps 7 to 11.  ||J||_2 = max_u Var_p(u), and the bound holds.
worst_ratio = 0.0
for _ in range(3000):
    n = int(rng.integers(2, 12))
    z = rng.normal(scale=rng.uniform(0.05, 8.0), size=n)
    p = softmax(z)
    J = np.diag(p) - np.outer(p, p)
    lam = float(np.linalg.eigvalsh(J).max())
    # step 9: the spectral norm is the largest variance of a unit-norm score
    u = np.linalg.eigh(J)[1][:, -1]
    var_p = float((p * u ** 2).sum() - (p * u).sum() ** 2)
    assert abs(lam - var_p) < 1e-12
    zs = np.sort(z)[::-1]
    bound = 4 * (n - 1) * np.exp(-(zs[0] - zs[1]))
    assert lam <= bound * (1 + 1e-9), (n, lam, bound)
    worst_ratio = max(worst_ratio, lam / bound)
assert worst_ratio > 0.4, "the bound is not vacuous: it is attained to within 2x"
print(f"||J||_2 <= 4(n-1)exp(-gap) over 3000 draws; tightest case"
      f" {worst_ratio:.3f} of the bound")

# The saturation the scaling exists to avoid: a gap of 20 nats kills the gradient.
z_big = np.array([20.0, 0.0, 0.0, 0.0])
p_big = softmax(z_big)
J_big = np.diag(p_big) - np.outer(p_big, p_big)
assert np.linalg.eigvalsh(J_big).max() < 1e-8
z_small = z_big / np.sqrt(MODEL_D.d_h)
p_small = softmax(z_small)
assert np.linalg.eigvalsh(np.diag(p_small) - np.outer(p_small, p_small)).max() > 1e-2

# The measurement: one trained layer, committed.  The hypotheses of part (a) do
# not hold in it, and the notebook reports by how much.
z_meas = np.load(find_data("ch11_scores.npz"))
Qm = z_meas["Q"].astype(np.float64)
Km = z_meas["K"].astype(np.float64)
Sm = z_meas["scores"].astype(np.float64)
h_m, s_m, dh_m = Qm.shape
recon = np.einsum("hid,hjd->hij", Qm, Km) / np.sqrt(dh_m)
assert np.abs(recon - Sm).max() < 1e-4, "the committed scores ARE QK^T/sqrt(d_h)"

for i in range(h_m):
    iid_prediction = np.sqrt(Qm[i].var() * Km[i].var() * dh_m) / np.sqrt(dh_m)
    realised = Sm[i].std()
    assert realised > 2.0 * iid_prediction, (i, realised, iid_prediction)
    C = np.corrcoef(Qm[i].T)
    off = ~np.eye(dh_m, dtype=bool)
    assert np.abs(C[off]).mean() > 0.2, "the coordinates of q are far from independent"
    assert np.abs(C[off]).max() > 0.8
print(f"trained layer: logit sd {Sm.std(axis=(1,2)).round(2)},"
      f" i.i.d. prediction ~{np.sqrt(Qm[0].var()*Km[0].var()):.2f}")
print(f"mean |correlation| between coordinates of q:"
      f" {np.abs(np.corrcoef(Qm[0].T)[~np.eye(dh_m, dtype=bool)]).mean():.3f}")"""

MD3 = """\
Concatenate-then-project is definitionally a sum, and the content of D-3.3 is
noticing that partitioning the output projection by rows turns h heads into h
low-rank additive writes into one shared space. The cell builds multi-head
attention the usual way, then rebuilds it as that sum, and the two agree to
machine precision. The rank of each OV circuit is checked directly, and the last
block separates multi-head from one wide head at identical parameter count: the
difference is h softmaxes rather than one, which is a nonlinearity and not a
bookkeeping choice."""

CODE3 = """\
from arith.model_d import MODEL_D, attention_params

def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

rng = np.random.default_rng(33)
s, d, h, d_h = 7, 16, 4, 4
assert d == h * d_h
X = rng.normal(size=(s, d))
W_Q = rng.normal(size=(d, h * d_h)) / np.sqrt(d)
W_K = rng.normal(size=(d, h * d_h)) / np.sqrt(d)
W_V = rng.normal(size=(d, h * d_h)) / np.sqrt(d)
W_O = rng.normal(size=(h * d_h, d)) / np.sqrt(h * d_h)

Q, K, V = X @ W_Q, X @ W_K, X @ W_V
heads, As = [], []
for i in range(h):
    sl = slice(i * d_h, (i + 1) * d_h)
    A_i = softmax(Q[:, sl] @ K[:, sl].T / np.sqrt(d_h))
    As.append(A_i)
    heads.append(A_i @ V[:, sl])

# Step 1 to 3: concatenate, then project.
C = np.concatenate(heads, axis=1)
assert C.shape == (s, h * d_h)
mha = C @ W_O

# Step 4: the same thing written as a sum of per-head additive writes.
W_OV = [W_V[:, i * d_h:(i + 1) * d_h] @ W_O[i * d_h:(i + 1) * d_h, :] for i in range(h)]
additive = sum(As[i] @ X @ W_OV[i] for i in range(h))
assert np.abs(mha - additive).max() < 1e-12, np.abs(mha - additive).max()

# Step 5: each OV circuit factors through R^{d_h}, so its rank is capped there.
for i in range(h):
    assert W_OV[i].shape == (d, d)
    assert np.linalg.matrix_rank(W_OV[i], tol=1e-10) <= d_h
    assert np.linalg.matrix_rank(W_OV[i], tol=1e-10) == d_h        # generically tight
# and so does the QK circuit
for i in range(h):
    sl = slice(i * d_h, (i + 1) * d_h)
    assert np.linalg.matrix_rank(W_Q[:, sl] @ W_K[:, sl].T, tol=1e-10) == d_h

# Step 6, at Model D's shapes: one head writes into 1/32 of the stream.
assert MODEL_D.d_h / MODEL_D.d == 1 / 32
assert MODEL_D.h * MODEL_D.d_h == MODEL_D.d

# Step 7: one wide head has exactly the same parameters and is a different
# function, because it applies one softmax where multi-head applies h.
wide = softmax(Q @ K.T / np.sqrt(h * d_h)) @ V @ W_O
assert wide.shape == mha.shape
assert np.abs(wide - mha).max() > 0.1, "same parameter count, different function"
n_params = W_Q.size + W_K.size + W_V.size + W_O.size
assert n_params == 4 * d * h * d_h

# The parameter count with grouped-query attention, from arith/, which is the
# same partition with fewer key and value blocks.
ap = attention_params(MODEL_D)
assert ap["W_Q"] == ap["W_O"] == MODEL_D.d * MODEL_D.h * MODEL_D.d_h
assert ap["W_K"] == MODEL_D.d * MODEL_D.n_kv * MODEL_D.d_h
assert sum(ap.values()) == 2 * MODEL_D.d * MODEL_D.d_h * (MODEL_D.h + MODEL_D.n_kv)
print(f"max |concat-then-project - sum of additive writes| ="
      f" {np.abs(mha - additive).max():.2e}")
print(f"h = {h} heads, each rank {np.linalg.matrix_rank(W_OV[0], tol=1e-10)}"
      f" of width {d}; one wide head differs by {np.abs(wide - mha).max():.3f}")"""

MD4 = """\
One row of attention is the Nadaraya-Watson estimator of 1964, and writing it
that way makes two things fall out. The first is the exponential factorisation
in step 3, which separates a genuine distance kernel from a per-key prior
proportional to the exponential of the key's squared norm, and that prior is the
attention-sink mechanism two sections before the chapter names it. The second is
that a kernel which factorises through a feature map turns the quadratic form
into two running sums, which is checked here both in the parallel form and as
the causal prefix-sum recurrence that Chapter 11 picks up."""

CODE4 = """\
rng = np.random.default_rng(34)
s, d_h, m = 9, 8, 6
q = rng.normal(size=d_h)
K = rng.normal(size=(s, d_h))
V = rng.normal(size=(s, d_h))
r = np.sqrt(d_h)

def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

# Steps 1 and 2: one attention row IS the Nadaraya-Watson estimator.
w = softmax(K @ q / r)
kernel = np.exp(K @ q / r)
nw = (kernel[:, None] * V).sum(axis=0) / kernel.sum()
assert np.abs(w @ V - nw).max() < 1e-12
assert abs(w.sum() - 1.0) < 1e-15

# Step 3: the exponential of an inner product factorises exactly.
lhs = np.exp(K @ q / r)
rhs = (np.exp(np.dot(q, q) / (2 * r)) * np.exp((K ** 2).sum(axis=1) / (2 * r))
       * np.exp(-((q - K) ** 2).sum(axis=1) / (2 * r)))
assert np.abs(lhs - rhs).max() / np.abs(lhs).max() < 1e-12

# Step 4: the query factor cancels in the normaliser, the key factor does not,
# so the weights are a Gaussian smoother reweighted by a per-key prior.
prior = np.exp((K ** 2).sum(axis=1) / (2 * r))
gauss = np.exp(-((q - K) ** 2).sum(axis=1) / (2 * r))
assert np.abs(w - (prior * gauss) / (prior * gauss).sum()).max() < 1e-12
assert prior.max() / prior.min() > 2.0, "the per-key prior spans more than a factor of two"

# The sink, constructed: two keys exactly equidistant from the query, differing
# only in norm.  The long one takes more of the mass, and by exactly the prior.
t = 0.6 * np.linalg.norm(q)
k_long = q * (1 + t / np.linalg.norm(q))
k_short = q * (1 - t / np.linalg.norm(q))
assert abs(np.linalg.norm(q - k_long) - np.linalg.norm(q - k_short)) < 1e-12
assert np.linalg.norm(k_long) > np.linalg.norm(k_short)
pair = softmax(np.stack([k_long, k_short]) @ q / r)
assert pair[0] > pair[1], "equal distance, unequal norm, unequal attention"
ratio_prior = np.exp((np.dot(k_long, k_long) - np.dot(k_short, k_short)) / (2 * r))
assert abs(pair[0] / pair[1] - ratio_prior) < 1e-10

# Step 5: impose QK-norm and the prior is exactly constant, leaving a Gaussian
# radial basis kernel with sigma^2 = sqrt(d_h).
qn = r * q / np.linalg.norm(q)
Kn = r * K / np.linalg.norm(K, axis=1, keepdims=True)
w_qk = softmax(Kn @ qn / r)
rbf = np.exp(-((qn - Kn) ** 2).sum(axis=1) / (2 * r))
assert np.abs(w_qk - rbf / rbf.sum()).max() < 1e-14
assert abs(np.linalg.norm(qn) - r) < 1e-12 and np.allclose(
    np.linalg.norm(Kn, axis=1), r, atol=1e-12)

# Step 6: softmax is a choice.  A different kernel is a different, valid smoother.
box = (np.abs(((q - K) ** 2).sum(axis=1)) < np.median(((q - K) ** 2).sum(axis=1))) * 1.0
box = box / box.sum()
assert abs(box.sum() - 1.0) < 1e-15
assert np.abs(box @ V - w @ V).max() > 1e-3, "the kernel changes the output"

# Steps 7 and 8: a factorising kernel makes the s x s matrix unnecessary.
Wf = rng.normal(size=(d_h, m)) / np.sqrt(d_h)
phi = lambda Z: np.exp(Z @ Wf)                     # positive, so no zero denominator
Qs = rng.normal(size=(s, d_h))
PQ, PK = phi(Qs), phi(K)
quad = (PQ @ PK.T) @ V / (PQ @ PK.T).sum(axis=1, keepdims=True)
S_state = PK.T @ V
z_state = PK.sum(axis=0)
lin = (PQ @ S_state) / (PQ @ z_state)[:, None]
assert np.abs(quad - lin).max() < 1e-12
assert S_state.shape == (m, d_h) and z_state.shape == (m,)
assert (PQ > 0).all() and (PK > 0).all()

# Step 9: causally, the two sums are prefix sums, which is a linear RNN.
S_run = np.zeros((m, d_h))
z_run = np.zeros(m)
rec = np.zeros((s, d_h))
for i in range(s):
    S_run = S_run + np.outer(PK[i], V[i])
    z_run = z_run + PK[i]
    rec[i] = PQ[i] @ S_run / (PQ[i] @ z_run)
mask = np.tril(np.ones((s, s)))
Wm = (PQ @ PK.T) * mask
causal = (Wm @ V) / Wm.sum(axis=1, keepdims=True)
assert np.abs(rec - causal).max() < 1e-12
assert S_run.shape == (m, d_h), "the state has fixed size, whatever s is"
print(f"max |attention row - Nadaraya-Watson| = {np.abs(w @ V - nw).max():.2e}")
print(f"per-key prior spans {prior.max()/prior.min():.2f}x;"
       f" under QK-norm it is constant")
print(f"linear form matches the quadratic one to {np.abs(quad - lin).max():.2e}")"""

SECTIONS = [
    ("1", "Attention as differentiable retrieval", MD1, CODE1),
    ("2", "The 1/sqrt(d_h) scaling", MD2, CODE2),
    ("3", "Multi-head attention is h rank-<=d_h additive writes", MD3, CODE3),
    ("4", "Attention is Nadaraya-Watson kernel smoothing", MD4, CODE4),
]
