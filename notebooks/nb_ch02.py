"""Chapter 2 — Embeddings as Geometry.

Four sections.  §2, §3 and §4 are the chapter's three `\\repo` margin notes,
D-2.1 through D-2.3; §1 is the tokenizer arithmetic and the size of the table
that D-2.1 differentiates, taken from `arith/model_d.py` rather than from
Table 1.2.
"""
from __future__ import annotations

CHAPTER = 2
SLUG = "embeddings"
TITLE = "Embeddings as Geometry"
BLURB = ("Byte-pair encoding as compression, the lookup as an exact matrix "
         "product with a scatter-add gradient, what cosine similarity throws "
         "away, and why two random directions in a wide space are almost "
         "always perpendicular.")

MD1 = """\
Byte-pair encoding is compression run once before training, and the vocabulary
size is an accounting identity rather than a design insight: 256 byte tokens
plus one token per merge plus the reserved slots. The cell below trains a small
byte-level BPE on a fixed corpus and checks three things the chapter claims,
that the count is exactly 256 plus the number of merges, that no input can be
out of vocabulary because the byte floor is always available, and that the merge
list is an ordered sequence rather than a set. It then sizes the table those ids
index into, at Model D's vocabulary and width, straight from `arith/`."""

CODE1 = """\
from arith.model_d import MODEL_D, embedding, non_embedding, total_params

CORPUS = (b"the quick brown fox jumps over the lazy dog. "
          b"the quicker browner foxes jumped over the lazier dogs. "
          b"low lower lowest slow slower slowest flow flowing flown. "
          b"compression is not linguistics, and a merge is not a morpheme. ")

def bpe_apply(ids, rules):
    # rules are (a, b, new): merge adjacent a,b into new, in the order given.
    for a, b, new in rules:
        out, i = [], 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                out.append(new)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        ids = out
    return ids

def bpe_train(data, n_merges):
    ids = list(data)
    vocab = {i: bytes([i]) for i in range(256)}
    rules = []
    for r in range(n_merges):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break
        a, b = max(sorted(counts), key=counts.get)       # (2.1), deterministic ties
        new = 256 + r
        rules.append((a, b, new))
        vocab[new] = vocab[a] + vocab[b]
        ids = bpe_apply(ids, [(a, b, new)])
    return rules, vocab, ids

def bpe_decode(ids, vocab):
    return b"".join(vocab[i] for i in ids)

m = 24
rules, vocab, ids = bpe_train(CORPUS, m)

# (2.2): the vocabulary size is 256 + m, with no special tokens here.
assert len(rules) == m
assert len(vocab) == 256 + m
assert max(vocab) == 255 + m

# Encoding is the same rule list replayed, in training order.
assert bpe_apply(list(CORPUS), rules) == ids
assert bpe_decode(ids, vocab) == CORPUS
assert len(ids) < len(CORPUS), "the merges have to compress something"

# No input is out of vocabulary: the byte floor covers every possible string.
rng = np.random.default_rng(2)
junk = bytes(rng.integers(0, 256, size=512).tolist())
assert bpe_decode(bpe_apply(list(junk), rules), vocab) == junk

# The list is ordered.  Replaying the same merges, with the same ids, in the
# reverse order gives a valid tokenisation of the same string, and a different one.
reordered = bpe_apply(list(CORPUS), rules[::-1])
assert bpe_decode(reordered, vocab) == CORPUS
assert reordered != ids, "a tokenizer is a sequence, not a set"

# Model D's table, from arith/ and not from the page.
c = MODEL_D
assert embedding(c) == c.V * c.d
assert 2 * embedding(c) == 2 * c.V * c.d
share = 2 * embedding(c) / total_params(c)
assert abs(100 * share - 13.08) < 5e-3
assert abs(2 * embedding(c) / 1e9 - 1.051) < 5e-4       # bf16 bytes, one copy
assert non_embedding(c) + 2 * embedding(c) == total_params(c)
print(f"corpus {len(CORPUS)} bytes -> {len(ids)} tokens with {m} merges,"
      f" vocab {len(vocab)}")
print(f"Model D: V*d = {embedding(c):,} per copy, {100*share:.2f}% of"
      f" {total_params(c):,} parameters")"""

MD2 = """\
The lookup that follows tokenisation is an exact matrix product with a one-hot
matrix, and saying so is what lets the linear-layer gradient of D-1.4 apply with
no new work. The cell below builds the one-hot matrix explicitly, checks that
the product and the gather agree to the last bit, and then checks that the
gradient computed by transposing the one-hot matrix is the same array that a
scatter-add produces. The last block is the part that matters in practice: at
most s of the V rows are touched, and a repeated token accumulates rather than
overwrites, which is why the operation is an index add and not an index copy."""

CODE2 = """\
rng = np.random.default_rng(22)
V, d, s = 11, 5, 8
W_E = rng.normal(size=(V, d))
token_ids = np.array([3, 7, 3, 0, 9, 7, 3, 1])       # note the repeats
assert len(token_ids) == s

O = np.zeros((s, V))
O[np.arange(s), token_ids] = 1.0
assert (O.sum(axis=1) == 1).all(), "exactly one 1 per row"

# (2.3) and (2.4): the product IS the lookup, bit for bit, not approximately.
X = O @ W_E
assert np.array_equal(X, W_E[token_ids])

# D-1.4 with X <- O and W <- W_E, checked against a central difference.
M = rng.normal(size=(s, d))
loss = lambda W: float((M * (O @ W)).sum())
dX = M
dW_E = O.T @ dX
num = np.zeros_like(W_E)
eps = 1e-6
for i in range(V):
    for k in range(d):
        E = np.zeros_like(W_E)
        E[i, k] = eps
        num[i, k] = (loss(W_E + E) - loss(W_E - E)) / (2 * eps)
assert np.abs(dW_E - num).max() < 1e-8
assert dW_E.shape == W_E.shape                        # denominator layout again

# Step 5: the same array as a scatter-add, which is what a framework runs.
scatter = np.zeros_like(W_E)
np.add.at(scatter, token_ids, dX)
assert np.array_equal(dW_E, scatter)

# At most s of the V rows are touched, and exactly the ones that occurred.
touched = np.flatnonzero(np.abs(dW_E).sum(axis=1) > 0)
assert set(touched.tolist()) == set(token_ids.tolist())
assert len(touched) <= s and len(touched) == len(set(token_ids.tolist())) == 5
assert (V - len(touched)) == 6, "six rows get no gradient at all from this batch"

# Repeats accumulate.  Token 3 occurs at positions 0, 2 and 6.
where3 = np.flatnonzero(token_ids == 3)
assert np.allclose(dW_E[3], dX[where3].sum(axis=0), atol=1e-15)
assert not np.allclose(dW_E[3], dX[where3[-1]], atol=1e-8), "index_add_, not index_copy_"

# Step 6: there is no gradient with respect to O.  The boundary of the learned
# system is here, and the sparsity pattern is data, not a parameter.
frac = len(touched) / V
print(f"{len(touched)} of {V} rows touched ({100*frac:.0f}%);"
      f" row 3 accumulates {len(where3)} contributions")
print(f"max |scatter-add - O^T dX| = {np.abs(scatter - dW_E).max():.1e}")"""

MD3 = """\
Cosine similarity is an inner product with both magnitudes divided out, and the
identity in step 3 of D-2.2 makes visible exactly when that is free and when it
is not. The cell checks the invariance, then the squared-distance identity, then
the equal-norm collapse under which cosine and Euclidean rankings must agree.
The counterexample asked for in E-2.4 is constructed here rather than asserted,
and the final block reproduces the mean-offset decomposition, which is the
mechanism behind a similarity matrix whose entries all sit near a positive
floor."""

CODE3 = """\
rng = np.random.default_rng(23)

def cos(x, y):
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

x = rng.normal(size=9)
y = rng.normal(size=9)

# Step 1: Cauchy-Schwarz bounds it, and the angle form is an identity.
theta = np.arccos(np.clip(cos(x, y), -1, 1))
assert abs(np.dot(x, y) - np.linalg.norm(x) * np.linalg.norm(y) * np.cos(theta)) < 1e-12
assert abs(cos(x, y)) <= 1 + 1e-15

# Step 2: invariance to positive rescaling of either argument, one degree of
# freedom per vector discarded.
for alpha, beta in [(0.01, 1.0), (1.0, 500.0), (3.7, 0.2)]:
    assert abs(cos(alpha * x, beta * y) - cos(x, y)) < 1e-12
assert abs(cos(-x, y) + cos(x, y)) < 1e-12          # and it is not sign-invariant

# Step 3: the squared-distance identity (2.7).
lhs = float(np.linalg.norm(x - y) ** 2)
rhs = (np.linalg.norm(x) ** 2 + np.linalg.norm(y) ** 2
       - 2 * np.linalg.norm(x) * np.linalg.norm(y) * cos(x, y))
assert abs(lhs - rhs) < 1e-12

# Step 4: at equal norms the two rankings agree exactly, over many draws.
r = 2.3
Z = rng.normal(size=(200, 9))
Z = r * Z / np.linalg.norm(Z, axis=1, keepdims=True)
q = r * (lambda v: v / np.linalg.norm(v))(rng.normal(size=9))
cs = Z @ q / (r * r)
ds = np.linalg.norm(Z - q, axis=1) ** 2
assert np.allclose(ds, 2 * r * r * (1 - cs), atol=1e-10)
assert np.array_equal(np.argsort(-cs), np.argsort(ds)), "same ranking, both ways"

# Step 5: unequal norms, and the rankings come apart.  E-2.4, in three dimensions.
a = np.array([1.0, 0.0, 0.0])
b = 3.0 * np.array([np.cos(np.deg2rad(10)), np.sin(np.deg2rad(10)), 0.0])
c_ = 1.0 * np.array([np.cos(np.deg2rad(40)), np.sin(np.deg2rad(40)), 0.0])
assert cos(a, b) > cos(a, c_)
assert np.linalg.norm(a - b) > np.linalg.norm(a - c_)
assert abs(cos(a, b) - np.cos(np.deg2rad(10))) < 1e-12

# (2.10): a common offset puts a floor under every cosine, and it is the same
# floor for every pair, so it carries no information about any pair.
mu = rng.normal(size=9)
mu = 4.0 * mu / np.linalg.norm(mu)
Xt = rng.normal(size=(300, 9))
Xc = mu + Xt
i, j = 5, 17
decomp = (np.dot(mu, mu) + np.dot(mu, Xt[i] + Xt[j]) + np.dot(Xt[i], Xt[j]))
assert abs(np.dot(Xc[i], Xc[j]) - decomp) < 1e-10
Xn = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
raw = Xn @ Xn.T
off = ~np.eye(300, dtype=bool)
Xd = Xc - Xc.mean(axis=0)
Xdn = Xd / np.linalg.norm(Xd, axis=1, keepdims=True)
cen = Xdn @ Xdn.T
assert raw[off].mean() > 0.5, "the offset dominates every raw cosine"
assert abs(cen[off].mean()) < 0.05, "centring removes the first two terms of (2.10)"
assert raw[off].mean() - cen[off].mean() > 0.5
print(f"mean off-diagonal cosine: raw {raw[off].mean():.3f},"
      f" mean-centred {cen[off].mean():.3f}")
print(f"cos(a,b) = {cos(a, b):.3f} at distance {np.linalg.norm(a-b):.3f};"
      f" cos(a,c) = {cos(a, c_):.3f} at distance {np.linalg.norm(a-c_):.3f}")"""

MD4 = """\
The statement of D-2.3 is a mean, a variance and a tail bound, and all three are
checked here by sampling rather than by restating the algebra. The variance
claim is the load-bearing one: the standard deviation of the cosine between two
independent random directions is one over the square root of the width, which at
Model D's width of 4096 is exactly one sixty-fourth. The last two blocks check
the tail bound empirically (the observed exceedance never beats the bound) and
the volume statement of step 7, that almost all of a ball in high dimension lies
in a thin shell at its surface."""

CODE4 = """\
from arith.model_d import MODEL_D

rng = np.random.default_rng(24)

def cosine_moments(d, n=20000, chunk=2500):
    total = n0 = 0.0
    sq = 0.0
    kept = []
    for _ in range(n // chunk):
        x = rng.standard_normal((chunk, d))
        y = rng.standard_normal((chunk, d))
        c = (np.einsum("ij,ij->i", x, y)
             / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)))
        total += c.sum()
        sq += (c ** 2).sum()
        n0 += chunk
        kept.append(c)
    return total / n0, sq / n0, np.concatenate(kept)

for d in (3, 64, 512, MODEL_D.d):
    mean, second, samples = cosine_moments(d)
    # Steps 3 and 4: mean zero, and E[cos^2] = 1/d exactly.
    assert abs(mean) < 4.0 / np.sqrt(d * 20000) + 1e-3
    assert abs(second * d - 1.0) < 0.05, (d, second * d)
    # Step 5: the standard deviation is 1/sqrt(d).
    assert abs(np.sqrt(second) - 1 / np.sqrt(d)) < 0.03 / np.sqrt(d)
    # Step 6: the tail bound is a bound, at every threshold tested.
    for eps in (0.5 / np.sqrt(d), 1.5 / np.sqrt(d), 4.0 / np.sqrt(d)):
        emp = float((np.abs(samples) >= eps).mean())
        assert emp <= 2 * np.exp(-d * eps ** 2 / 2) + 1e-12, (d, eps, emp)
    print(f"d = {d:>5}: mean {mean:+.5f}, sd {np.sqrt(second):.5f},"
          f" 1/sqrt(d) {1/np.sqrt(d):.5f}")

# At Model D's width the number is exact, not approximate: 4096 = 64^2.
assert MODEL_D.d == 4096 and np.sqrt(MODEL_D.d) == 64.0
assert 1 / np.sqrt(MODEL_D.d) == 0.015625
assert abs(1 / np.sqrt(MODEL_D.d) - 1 / 64) < 1e-17

# Step 7, equation (2.9): volume lives at the surface.
delta = 0.001
shell = 1 - (1 - delta) ** MODEL_D.d
assert abs(100 * shell - 98.3) < 0.05
assert 1 - (1 - delta) ** 3 < 0.01, "in three dimensions the same shell is empty"

# The floor the failure mode warns about: 3/sqrt(d) is where a similarity stops
# being indistinguishable from noise, and it is tiny at this width.
floor = 3 / np.sqrt(MODEL_D.d)
_, _, samples = cosine_moments(MODEL_D.d, n=20000)
assert float((np.abs(samples) >= floor).mean()) < 0.01
assert floor < 0.05
print(f"shell of relative thickness {delta} holds {100*shell:.1f}% of a"
      f" {MODEL_D.d}-dimensional ball")
print(f"noise floor 3/sqrt(d) = {floor:.4f}")"""

SECTIONS = [
    ("1", "Byte-pair encoding, and the size of the table", MD1, CODE1),
    ("2", "The lookup is a matrix product, and its gradient is a scatter-add",
     MD2, CODE2),
    ("3", "What cosine similarity is, and is not", MD3, CODE3),
    ("4", "Two random directions in R^d are almost orthogonal", MD4, CODE4),
]
