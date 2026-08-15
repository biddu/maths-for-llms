"""Chapter 4 — Position.

Five sections.  The chapter carries `\\repo` notes for §1, §2, §3 and §5, which
are D-4.1, D-4.2, D-4.3 and D-4.4; §4 fills the gap with the frequency ladder,
read out of `arith/model_d.py::rope_bands` so that the wavelengths, the critical
dimension and the band counts in the arithmetic box are recomputed rather than
quoted.
"""
from __future__ import annotations

CHAPTER = 4
SLUG = "position"
TITLE = "Position"
BLURB = ("Why a positional term is forced rather than chosen, why the solution "
         "of the relative-inner-product equation has to be a rotation, and what "
         "the frequency ladder implies when a model trained to 8192 tokens is "
         "asked for 131072.")

MD1 = """\
Permutation equivariance is two lines of algebra with a rider that decides the
whole chapter. The cell below checks the statement on unmasked attention, where
it holds to machine precision, and then checks the rider: the causal mask is not
invariant under a simultaneous permutation of rows and columns, so a decoder is
not equivariant and the mask alone leaks order. The last block makes precise
what the mask does not supply, which is a metric. It distinguishes before from
after and gives no way at all to tell a gap of three from a gap of three
thousand."""

CODE1 = """\
def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

rng = np.random.default_rng(41)
s, d, d_h = 6, 10, 5
X = rng.normal(size=(s, d))
W_Q = rng.normal(size=(d, d_h)) / np.sqrt(d)
W_K = rng.normal(size=(d, d_h)) / np.sqrt(d)
W_V = rng.normal(size=(d, d_h)) / np.sqrt(d)

def attn(X, mask=None):
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V          # step 1: projections act on the right
    S = Q @ K.T / np.sqrt(d_h)                   # step 2
    if mask is not None:
        S = S + mask
    return softmax(S) @ V                        # steps 3 to 5

perm = rng.permutation(s)
P = np.eye(s)[perm]
assert np.array_equal(P @ X, X[perm])
assert np.allclose(P.T @ P, np.eye(s), atol=1e-15)          # step 5

# The statement: Attn(PX) = P Attn(X), exactly.
assert np.abs(attn(P @ X) - P @ attn(X)).max() < 1e-13

# Step 3, on its own: a row-wise softmax commutes with a simultaneous permutation.
S = (X @ W_Q) @ (X @ W_K).T / np.sqrt(d_h)
assert np.abs(softmax(P @ S @ P.T) - P @ softmax(S) @ P.T).max() < 1e-15
assert abs(np.exp(S[0]).sum() - np.exp(S[0][perm]).sum()) < 1e-12, "reordered summands"

# The rider.  The causal mask does not satisfy P M P^T = M, so equivariance goes.
NEG = -1e9
M = np.triu(np.full((s, s), NEG), 1)
assert np.abs(P @ M @ P.T - M).max() > 1.0
assert not np.allclose(attn(P @ X, M), P @ attn(X, M), atol=1e-6)
assert np.abs(attn(P @ X, M) - P @ attn(X, M)).max() > 1e-4

# What the mask does supply: a count.  Row t of a causal attention matrix has
# exactly t+1 unmasked entries, which is a monotone function of position.
A = softmax(S + M)
visible = (A > 1e-12).sum(axis=1)
assert list(visible) == list(range(1, s + 1))

# What it does not supply: a metric.  Swap two keys that are both in the past of
# a query and the query's output does not change at all.
Xg = X.copy()
Xs = X.copy()
Xs[[1, 3]] = Xs[[3, 1]]
Ag = softmax((Xg @ W_Q) @ (Xg @ W_K).T / np.sqrt(d_h) + M)[5]
As_ = softmax((Xs @ W_Q) @ (Xs @ W_K).T / np.sqrt(d_h) + M)[5]
assert np.abs(np.sort(Ag) - np.sort(As_)).max() < 1e-12, "same weights, relabelled"
assert np.abs(attn(Xg, M)[5] - attn(Xs, M)[5]).max() < 1e-12
print(f"unmasked: max |Attn(PX) - P Attn(X)| = {np.abs(attn(P@X) - P@attn(X)).max():.2e}")
print(f"causal:   max |Attn(PX) - P Attn(X)| ="
      f" {np.abs(attn(P@X, M) - P@attn(X, M)).max():.3f}")"""

MD2 = """\
Equation (4.4) is a functional equation, and D-4.2 solves it rather than
guessing. Restricting to maps linear in x, orthogonality is forced at step 2
(not assumed), the one-parameter group property follows at step 3, and the
canonical form of a skew-symmetric generator gives the block rotations at step
4. Each of those steps is checked below as a separate claim, including the
matrix exponential, which is compared against the block-diagonal rotation built
directly. The last block is the failure mode: drop orthogonality and the logit
acquires a term that depends on absolute position, which is what breaks low-rank
absorption in Chapter 11."""

CODE2 = """\
from scipy.linalg import expm

rng = np.random.default_rng(42)
d_h = 8
n_pairs = d_h // 2
base = 10000.0
theta = base ** (-2 * np.arange(n_pairs) / d_h)          # (4.9)

def R(m):
    A = np.zeros((d_h, d_h))
    for i, t in enumerate(theta):
        a = m * t
        c, s_ = np.cos(a), np.sin(a)
        A[2 * i, 2 * i] = c
        A[2 * i, 2 * i + 1] = -s_
        A[2 * i + 1, 2 * i] = s_
        A[2 * i + 1, 2 * i + 1] = c
    return A

q = rng.normal(size=d_h)
k = rng.normal(size=d_h)

# Step 2: A_0 = I, and orthogonality is forced by setting n = m.
assert np.allclose(R(0), np.eye(d_h), atol=1e-15)
for m in (0, 1, 3, 17, 4096):
    assert np.allclose(R(m).T @ R(m), np.eye(d_h), atol=1e-10)
    assert abs(np.linalg.norm(R(m) @ q) - np.linalg.norm(q)) < 1e-10
    assert abs(np.linalg.det(R(m)) - 1.0) < 1e-10

# Step 3: one-parameter group.  A_m = A_1^m, and A_m^{-1} A_n = A_{n-m}.
assert np.allclose(np.linalg.matrix_power(R(1), 5), R(5), atol=1e-12)
for (m, n) in [(0, 4), (3, 7), (11, 15), (100, 104)]:
    assert np.allclose(np.linalg.inv(R(m)) @ R(n), R(n - m), atol=1e-9)

# Steps 3 and 4: the generator is skew-symmetric and the exponential reproduces
# the block rotations exactly.
Theta = np.zeros((d_h, d_h))
for i, t in enumerate(theta):
    Theta[2 * i, 2 * i + 1] = -t
    Theta[2 * i + 1, 2 * i] = t
assert np.allclose(Theta, -Theta.T, atol=1e-15)
for m in (1, 2, 9):
    assert np.abs(expm(m * Theta) - R(m)).max() < 1e-10

# Step 5: the pairing of coordinates is a change of basis, not a constraint.
# Conjugating by any orthogonal Q gives another solution of the same equation.
G = rng.normal(size=(d_h, d_h))
Qo = np.linalg.qr(G)[0]
Rot = lambda m: Qo @ R(m) @ Qo.T
for (m, n) in [(2, 6), (5, 9)]:
    assert np.allclose(Rot(m).T @ Rot(n), Rot(n - m), atol=1e-9)
    assert abs((Rot(m) @ q) @ (Rot(n) @ k) - q @ Rot(n - m) @ k) < 1e-9
assert np.abs(Rot(3) - R(3)).max() > 0.1, "a different basis, the same solution family"

# The requirement itself, (4.4): the logit depends on m and n only through n - m.
for offset in (0, 1, 5):
    vals = [(R(m) @ q) @ (R(m + offset) @ k) for m in (0, 1, 2, 40, 4000)]
    assert max(vals) - min(vals) < 1e-9, offset

# Step 6, at d_h = 2, in complex form: only the difference of angles survives.
q2 = np.array([1.3, -0.4])
k2 = np.array([0.7, 0.9])
zq = complex(*q2)
zk = complex(*k2)
th = 0.37
for (m, n) in [(0, 3), (2, 5), (10, 13)]:
    Rm = np.array([[np.cos(m * th), -np.sin(m * th)], [np.sin(m * th), np.cos(m * th)]])
    Rn = np.array([[np.cos(n * th), -np.sin(n * th)], [np.sin(n * th), np.cos(n * th)]])
    lhs = (Rm @ q2) @ (Rn @ k2)
    rhs = abs(zq) * abs(zk) * np.cos(np.angle(zq) - np.angle(zk) + (m - n) * th)
    assert abs(lhs - rhs) < 1e-12

# The failure mode: drop orthogonality, keep everything else.  A_m = g^m R(m)
# solves nothing, because the logit now depends on m + n.
g = 1.1
A_bad = lambda m: (g ** m) * R(m)
vals = [(A_bad(m) @ q) @ (A_bad(m + 4) @ k) for m in (0, 3, 10)]
assert max(vals) - min(vals) > 1.0, "absolute position has leaked into the logit"
assert abs(np.linalg.norm(A_bad(10) @ q) / np.linalg.norm(q) - g ** 10) < 1e-9
print(f"orthogonality holds to {np.abs(R(4096).T @ R(4096) - np.eye(d_h)).max():.1e}")
print(f"expm(m*Theta) matches the block form to {np.abs(expm(9*Theta) - R(9)).max():.1e}")
print(f"non-orthogonal A_m: logit at fixed offset 4 spans {max(vals)-min(vals):.2f}")"""

MD3 = """\
The shift identity is what makes RoPE cheap: rotating queries and keys by their
absolute positions produces a logit that only ever sees the difference, so no
relative-position matrix has to be materialised. The cell verifies the identity
block by block and then whole, checks the logit form both ways, and reproduces
step 5's per-block expansion. Note that the sine coefficient is antisymmetric,
so its sign tracks whether the rotation is applied on the left (column
convention, as in the statement of D-4.3) or on the right (the book's row
convention, as in step 5); both forms are asserted here. The last block is the
failure mode, that the oscillation is bounded and undecaying, so RoPE supplies
no recency prior at all."""

CODE3 = """\
rng = np.random.default_rng(43)
d_h = 16
n_pairs = d_h // 2
base = 500_000.0
theta = base ** (-2 * np.arange(n_pairs) / d_h)

def R(m):
    A = np.zeros((d_h, d_h))
    for i, t in enumerate(theta):
        a = m * t
        c, s_ = np.cos(a), np.sin(a)
        A[2 * i, 2 * i] = c
        A[2 * i, 2 * i + 1] = -s_
        A[2 * i + 1, 2 * i] = s_
        A[2 * i + 1, 2 * i + 1] = c
    return A

q = rng.normal(size=d_h)
k = rng.normal(size=d_h)

# Steps 1 to 3: R(a)^T = R(-a), R(-a)R(b) = R(b-a), and it lifts block by block.
a, b = 0.31, 1.27
Ra = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
Rb = np.array([[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]])
Rba = np.array([[np.cos(b - a), -np.sin(b - a)], [np.sin(b - a), np.cos(b - a)]])
assert np.abs(Ra.T @ Rb - Rba).max() < 1e-15
for (m, n) in [(0, 1), (3, 7), (250, 4000), (8191, 8191)]:
    assert np.abs(R(m).T @ R(n) - R(n - m)).max() < 1e-9, (m, n)

# Step 4: the logit, both ways.
for (m, n) in [(0, 5), (12, 3), (100, 4100)]:
    lhs = (R(m) @ q) @ (R(n) @ k)
    assert abs(lhs - q @ R(n - m) @ k) < 1e-9
    assert abs(lhs - (R(m) @ q) @ (R(n) @ k)) < 1e-15

# Step 5, per block.  Column convention (rotation on the left) and the book's row
# convention (rotation on the right) differ only in the sign of the sine term,
# because that term is antisymmetric in q and k.
m, n = 7, 19
col = q @ R(n - m) @ k
row = (q @ R(m)) @ (k @ R(n))
by_block_col = sum(
    (q[2 * i] * k[2 * i] + q[2 * i + 1] * k[2 * i + 1]) * np.cos((n - m) * t)
    + (q[2 * i + 1] * k[2 * i] - q[2 * i] * k[2 * i + 1]) * np.sin((n - m) * t)
    for i, t in enumerate(theta))
by_block_row = sum(
    (q[2 * i] * k[2 * i] + q[2 * i + 1] * k[2 * i + 1]) * np.cos((n - m) * t)
    + (q[2 * i] * k[2 * i + 1] - q[2 * i + 1] * k[2 * i]) * np.sin((n - m) * t)
    for i, t in enumerate(theta))
assert abs(col - by_block_col) < 1e-12
assert abs(row - by_block_row) < 1e-12
assert abs(col - row) > 1e-6, "the two conventions are genuinely different numbers"

# The whole point: no s x s relative-position matrix is ever built.  The logit
# matrix of a sequence depends only on the difference of indices, so it is
# Toeplitz before masking.
s_len = 24
Qs = rng.normal(size=(s_len, d_h))
Ks = rng.normal(size=(s_len, d_h))
Qr = np.stack([R(m) @ Qs[m] for m in range(s_len)])
Kr = np.stack([R(n) @ Ks[n] for n in range(s_len)])
Zr = Qr @ Kr.T
direct = np.array([[Qs[m] @ R(n - m) @ Ks[n] for n in range(s_len)]
                   for m in range(s_len)])
assert np.abs(Zr - direct).max() < 1e-9

# With a single shared q and k, the logit is a function of n - m alone, which is
# exactly what Toeplitz means.
Zc = np.array([[(R(m) @ q) @ (R(n) @ k) for n in range(s_len)] for m in range(s_len)])
for off in range(-s_len + 1, s_len):
    diag = np.diagonal(Zc, offset=off)
    assert diag.max() - diag.min() < 1e-9

# The failure mode: bounded oscillation, no decay.  A key 4000 tokens away is
# drawn from the same range as one 4 tokens away.
near = np.array([q @ R(t) @ k for t in range(1, 40)])
far = np.array([q @ R(t) @ k for t in range(4000, 4039)])
bound = np.linalg.norm(q) * np.linalg.norm(k)
assert np.abs(near).max() <= bound + 1e-9 and np.abs(far).max() <= bound + 1e-9
assert np.abs(far).max() > 0.25 * np.abs(near).max(), "no decay with distance"
print(f"max |R_m^T R_n - R_(n-m)| over the tested pairs = "
      f"{np.abs(R(250).T @ R(4000) - R(3750)).max():.1e}")
print(f"logit range near {np.abs(near).max():.3f} vs far {np.abs(far).max():.3f},"
      f" Cauchy-Schwarz bound {bound:.3f}")"""

MD4 = """\
Reading the frequencies as wavelengths is what turns the ladder into a
diagnostic. The cell recomputes the ladder from Model D's own hyperparameters,
checks it against `arith/model_d.py::rope_bands`, and then locates the critical
dimension two ways, as the first index whose wavelength exceeds the trained
context and by the closed form in equation (4.13). The last block is the reason
the base moved from ten thousand to five hundred thousand, which is a
precondition for long context rather than a tuning choice: at the lower base no
pair has a wavelength long enough to separate positions 131072 apart."""

CODE4 = """\
import math
from dataclasses import replace
from arith.model_d import MODEL_D, rope_bands, critical_dimension

c = MODEL_D
n_pairs = c.d_h // 2
i_idx = np.arange(n_pairs)

# (4.9) and (4.10): frequencies and wavelengths, from the hyperparameters.
theta = c.rope_base ** (-2 * i_idx / c.d_h)
lam = 2 * np.pi / theta
assert np.allclose(lam, 2 * np.pi * c.rope_base ** (2 * i_idx / c.d_h), atol=1e-6)
assert np.allclose(theta * lam, 2 * np.pi, atol=1e-12)

rows = rope_bands(c)
assert len(rows) == n_pairs == 64
assert np.allclose([r["lambda"] for r in rows], lam, rtol=1e-12)
assert abs(lam[0] - 2 * np.pi) < 1e-12 and abs(lam[0] - 6.28) < 5e-3
assert abs(lam[-1] / 1e6 - 2.56) < 5e-3
assert abs(2 * np.pi * c.rope_base / 1e6 - 3.14) < 5e-3      # the ceiling
assert lam[-1] < 2 * np.pi * c.rope_base

# Turns completed over the trained context, which is the diagnostic.
turns = c.trained_context / lam
assert np.allclose([r["turns"] for r in rows], turns, rtol=1e-12)

# (4.13): the critical dimension, two ways.
i_star = int(np.flatnonzero(lam > c.trained_context)[0])
closed = math.ceil(c.d_h / 2 * math.log(c.trained_context / (2 * math.pi))
                   / math.log(c.rope_base))
assert i_star == closed == critical_dimension(c) == 35
assert lam[i_star - 1] < c.trained_context < lam[i_star]
assert abs(lam[34] - 6695) < 1.0 and abs(lam[35] - 8219) < 1.0
assert abs(turns[34] - 1.22) < 5e-3 and abs(turns[35] - 0.997) < 5e-4

# The fraction of pairs that complete a turn during training.
frac = i_star / n_pairs
assert abs(100 * frac - 54.7) < 0.05

# Why the base moved.  At b = 10000 the ceiling is far below the extended
# context, so no pair can separate positions 131072 apart without aliasing.
low = replace(c, rope_base=10_000)
lam_low = 2 * np.pi * low.rope_base ** (2 * i_idx / low.d_h)
assert np.allclose([r["lambda"] for r in rope_bands(low)], lam_low, rtol=1e-12)
assert 50e3 < lam_low[-1] < 2 * np.pi * low.rope_base       # tens of thousands of tokens
assert lam_low.max() < c.extended_context
assert lam.max() > c.extended_context
assert 2 * np.pi * low.rope_base < c.extended_context

# And what it costs: fewer pairs complete a turn during training at the higher
# base, which is the trade E-4.7 works both ways.
i_star_low = critical_dimension(low)
assert i_star_low > i_star, (i_star_low, i_star)
print(f"lambda_0 = {lam[0]:.2f} tokens, lambda_63 = {lam[-1]:.3e},"
      f" ceiling {2*np.pi*c.rope_base:.3e}")
print(f"critical dimension i* = {i_star} ({100*frac:.1f}% of pairs turn at least"
      f" once); at base 10,000 it is {i_star_low}")"""

MD5 = """\
The three extension methods are three policies over the one diagnostic of §4,
and once the turns are in hand they are easy to separate. The cell reconstructs
YaRN's ramp from Model D's hyperparameters and checks it against
`arith/model_d.py::rope_bands`, including the band counts the arithmetic box
prints, then checks the two limiting cases: setting the ramp to zero everywhere
recovers position interpolation exactly, and setting it to one everywhere is a
no-op. NTK-aware scaling is checked in the form the chapter states it, that the
slowest pair is stretched by exactly s and the fastest is left untouched. The
last block is the temperature, where the square-root convention and the version
without it differ by a factor that a reviewer who knows the paper will notice."""

CODE5 = """\
import math
from arith.model_d import MODEL_D, rope_bands, critical_dimension

c = MODEL_D
n_pairs = c.d_h // 2
i_idx = np.arange(n_pairs)
theta = c.rope_base ** (-2 * i_idx / c.d_h)
lam = 2 * np.pi / theta
s_scale = c.extended_context / c.trained_context
assert s_scale == 16.0

# The diagnostic, and YaRN's ramp built from it.  alpha = 1, beta = 32.
alpha_y, beta_y = 1.0, 32.0
r_i = c.trained_context / lam
gamma = np.clip((r_i - alpha_y) / (beta_y - alpha_y), 0.0, 1.0)
theta_yarn = (1 - gamma) * theta / s_scale + gamma * theta

rows = rope_bands(c)
assert np.allclose(gamma, [r["gamma"] for r in rows], atol=1e-12)
# rope_bands reports the effective scale, which is exactly theta / theta'.
assert np.allclose(theta / theta_yarn, [r["effective_scale"] for r in rows], rtol=1e-12)

# The three bands of the arithmetic box, counted rather than quoted.
n_extrapolate = int((gamma >= 1 - 1e-12).sum())
n_interpolate = int((gamma <= 1e-12).sum())
n_ramp = n_pairs - n_extrapolate - n_interpolate
assert (n_extrapolate, n_ramp, n_interpolate) == (19, 16, 29)
assert n_extrapolate + n_ramp + n_interpolate == n_pairs
# The bands are contiguous and ordered, fast pairs first.
assert (gamma[:n_extrapolate] >= 1 - 1e-12).all()
assert (gamma[n_pairs - n_interpolate:] <= 1e-12).all()
assert np.all(np.diff(gamma) <= 1e-15), "gamma is non-increasing in i"
# and the boundaries are exactly where the turns cross alpha and beta.
assert r_i[n_extrapolate - 1] >= beta_y > r_i[n_extrapolate]
assert r_i[n_pairs - n_interpolate] <= alpha_y
assert critical_dimension(c) == 35 and gamma[35] == 0.0

# Step 3: position interpolation is the gamma = 0 policy, applied to every band.
theta_pi = theta / s_scale
assert np.allclose(theta_pi, (1 - 0.0) * theta / s_scale + 0.0 * theta, atol=1e-18)
assert abs(theta[0] - 1.0) < 1e-12 and abs(theta_pi[0] - 1 / s_scale) < 1e-12
assert np.allclose(theta_yarn[gamma <= 1e-12], theta_pi[gamma <= 1e-12], rtol=1e-12)
# and the fastest bands, which YaRN leaves alone, are exactly where PI hurts.
assert np.allclose(theta_yarn[:n_extrapolate], theta[:n_extrapolate], rtol=1e-12)
assert (theta_pi[:n_extrapolate] < theta[:n_extrapolate]).all()

# Step 4: NTK-aware scaling raises the base instead.
base_ntk = c.rope_base * s_scale ** (c.d_h / (c.d_h - 2))
lam_ntk = 2 * np.pi * base_ntk ** (2 * i_idx / c.d_h)
assert abs(lam_ntk[-1] / lam[-1] - s_scale) < 1e-9, "the slowest pair stretches by s"
assert abs(lam_ntk[0] / lam[0] - 1.0) < 1e-12, "the fastest pair is untouched"
assert np.allclose(lam_ntk / lam, s_scale ** (2 * i_idx / (c.d_h - 2)), rtol=1e-12)
assert abs(base_ntk / 1e6 - 8.36) < 5e-3

# Step 6: the temperature.  The paper fits sqrt(1/t), and the difference between
# the two conventions is the same factor again, which is the point of the note.
root_form = 0.1 * math.log(s_scale) + 1
plain_form = root_form ** 2
assert abs(plain_form / root_form - root_form) < 1e-12
assert plain_form - root_form > 0.35, "a reviewer who knows the paper will catch this"
assert abs(root_form - 1.2773) < 5e-5 and abs(plain_form - 1.6314) < 5e-5
# Lower temperature means a sharper softmax, so the correction really does move
# the attention entropy in the direction step 6 claims.
def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()
rng = np.random.default_rng(45)
z = rng.normal(scale=2.0, size=64)
H = lambda p: float(-(p * np.log(p)).sum())
assert H(softmax(z / root_form)) > H(softmax(z))
assert H(softmax(z * root_form)) < H(softmax(z))

# The failure mode: using the already-extended length as L makes every band
# extrapolate, and the method silently does nothing.
r_wrong = c.extended_context / lam
gamma_wrong = np.clip((r_wrong - alpha_y) / (beta_y - alpha_y), 0.0, 1.0)
theta_wrong = (1 - gamma_wrong) * theta / s_scale + gamma_wrong * theta
assert int((gamma_wrong >= 1 - 1e-12).sum()) > n_extrapolate
assert np.abs(theta_wrong - theta).max() < np.abs(theta_yarn - theta).max()
print(f"bands: {n_extrapolate} extrapolate, {n_ramp} ramp, {n_interpolate} interpolate")
print(f"effective scale at i=19 {theta[19]/theta_yarn[19]:.2f},"
      f" at i=34 {theta[34]/theta_yarn[34]:.2f}, at i=35 {theta[35]/theta_yarn[35]:.2f}")
print(f"NTK base {base_ntk:.4e}; temperature sqrt(1/t) = {root_form:.4f}"
      f" vs 1/t = {plain_form:.4f}")"""

SECTIONS = [
    ("1", "Attention without position is permutation-equivariant", MD1, CODE1),
    ("2", "RoPE is the general solution of the relative-inner-product equation",
     MD2, CODE2),
    ("3", "The shift identity and the relative logit", MD3, CODE3),
    ("4", "The frequency ladder as wavelengths", MD4, CODE4),
    ("5", "The wavelength diagnostic and YaRN's ramp", MD5, CODE5),
]
