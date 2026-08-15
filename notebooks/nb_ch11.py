"""Chapter 11 — The Cost of Attention.

Generated into `notebooks/ch11_cost_of_attention.ipynb` by `build_all.py`.  The
chapter cites §1, §3, §4 and §5 by number, so sections may be added but never
renumbered.

§5 is the one to read carefully.  Online softmax is exact in exact arithmetic
and is never bitwise equal to an unblocked reference in floating point, so the
exactness is demonstrated over the rationals and the floating-point agreement
is asserted with a tolerance.  Asserting equality there would be asserting the
wrong thing.
"""
from __future__ import annotations

CHAPTER = 11
SLUG = "cost_of_attention"
TITLE = "The Cost of Attention"
BLURB = (
    "Decode is a bandwidth problem and prefill is a memory-layout problem. "
    "The cache formula, the arithmetic intensity that follows from it, the "
    "absorption identity and the rotation that obstructs it, and the online "
    "recurrence that makes the score matrix unnecessary."
)

S1 = r'''
from arith.accelerators import DEFAULT, GiB
from arith.kv_cache import (BF16, FP8, FP32, KiB, cache_bytes,
                            decode_flops, decode_intensity, latency_floor,
                            latent_bytes_per_token, per_head_bytes_per_token,
                            schemes)
from arith.model_d import MODEL_D
from arith.model_s import MODEL_S

d, s_ = MODEL_D, MODEL_S

# ---- steps 1 to 4.  The leading 2 is K-and-V, two tensors.  It is NOT bytes
# per element: that is p_b, and it happens also to equal 2 in bf16, which is
# the collision every wrong reproduction of (11.2) makes.
gqa = per_head_bytes_per_token(d.L, d.n_kv, d.d_h, BF16)
assert gqa == 2 * d.L * d.n_kv * d.d_h * 2
assert gqa == 131_072 == 128 * KiB
# change ONLY the precision and the cache halves; the K-and-V two is untouched
assert per_head_bytes_per_token(d.L, d.n_kv, d.d_h, FP8) == gqa // 2
assert per_head_bytes_per_token(d.L, d.n_kv, d.d_h, FP32) == 2 * gqa
# the error, sized: collapsing the two twos into one is a factor of exactly two
collapsed = d.L * d.n_kv * d.d_h * BF16
assert gqa / collapsed == 2.0
print("Model D GQA: %d bytes per token = 2 (K and V) x %d layers x %d kv "
      "heads x %d wide x %d bytes/element" % (gqa, d.L, d.n_kv, d.d_h, BF16))

# The ladder.  MQA, GQA and MHA are one formula at three values of one integer.
r = schemes()
assert r["D MHA"]["bytes"] == 512 * KiB and r["D MQA"]["bytes"] == 16 * KiB
assert r["D MHA"]["bytes"] / r["D GQA"]["bytes"] == d.h / d.n_kv == 4
assert r["D GQA"]["bytes"] / r["D MQA"]["bytes"] == d.n_kv == 8
# and the latent cache is a different formula, with no factor of two at all
assert latent_bytes_per_token(s_.L, s_.d_c, s_.d_r, BF16) == (s_.d_c + s_.d_r) * s_.L * 2
assert r["S MLA"]["bytes"] == 70_272 and round(r["S MLA"]["bytes"] / KiB, 3) == 68.625
assert round(r["S MHA"]["bytes"] / r["S MLA"]["bytes"], 2) == 56.89
print("cache per token: D MHA %.0f KiB, D GQA %.0f KiB, D MQA %.0f KiB, "
      "S MLA %.3f KiB" % (r["D MHA"]["bytes"] / KiB, gqa / KiB,
                          r["D MQA"]["bytes"] / KiB, r["S MLA"]["bytes"] / KiB))

# It is linear in s, so the cache is a running cost and not a fixed overhead.
for n in (2, 4, 16):
    assert cache_bytes(gqa, 8192 * n) == n * cache_bytes(gqa, 8192)
assert cache_bytes(gqa, 131_072) / GiB == 16.0
assert cache_bytes(gqa, 131_072, b=4) == 4 * cache_bytes(gqa, 131_072)
print("at s = %d the GQA cache is %.0f GiB for ONE sequence, and it does not "
      "amortise across a batch" % (131_072, cache_bytes(gqa, 131_072) / GiB))

# ---- steps 5 to 7.  L, d_h and s cancel identically, so the intensity is the
# ratio of query heads to key/value heads and the precision, and nothing else.
for s in (128, 1024, 8192, 131_072):
    for n_kv in (1, 2, 8, 32):
        for p_b in (BF16, FP8):
            flops = decode_flops(d.L, d.h, d.d_h, s)
            byts = per_head_bytes_per_token(d.L, n_kv, d.d_h, p_b) * s
            assert abs(flops / byts - decode_intensity(d.h, n_kv, p_b)) < 1e-12
# and it does not depend on the depth or the head width either
for L in (8, 32, 61):
    for d_h in (64, 128, 256):
        f2 = decode_flops(L, d.h, d_h, 4096)
        b2 = per_head_bytes_per_token(L, d.n_kv, d_h, BF16) * 4096
        assert abs(f2 / b2 - 4.0) < 1e-12, (L, d_h)
print("intensity is 2h/(n_kv p_b) whatever s, L and d_h are: checked over "
      "%d combinations" % (4 * 4 * 2 + 3 * 3))

# ---- step 8, substituted.
assert decode_intensity(d.h, d.n_kv, BF16) == 4.0
assert decode_intensity(d.h, d.h, BF16) == 1.0        # the same model with MHA
assert decode_intensity(d.h, 1, BF16) == 32.0         # and with MQA
assert decode_intensity(d.h, d.n_kv, FP8) == 8.0      # halving p_b doubles it

# ---- step 9, machine balance.  One hardware number, and it is the only one.
assert round(DEFAULT.balance, 1) == 295.2
assert round(DEFAULT.balance / decode_intensity(d.h, d.n_kv, BF16), 1) == 73.8
print("machine balance %.1f FLOP/byte against an intensity of %.0f: decode "
      "attention sits %.1fx below the ridge, so the FLOP count never enters"
      % (DEFAULT.balance, decode_intensity(d.h, d.n_kv),
         DEFAULT.balance / decode_intensity(d.h, d.n_kv)))

# The floor that follows, and it is a floor no kernel work can lift.
b = cache_bytes(gqa, 131_072)
assert round(1e3 * latency_floor(b), 2) == 5.13
assert round(1 / latency_floor(b)) == 195
# halving the arithmetic halves nothing, because the time is bytes/bandwidth
assert latency_floor(b) == b / DEFAULT.bandwidth
print("reading %.0f GiB at %.2f TB/s takes %.2f ms, so at most %.0f tokens/s "
      "at b = 1, before a single weight byte is touched"
      % (b / GiB, DEFAULT.bandwidth / 1e12, 1e3 * latency_floor(b),
         1 / latency_floor(b)))
'''

S2 = r'''
from arith.accelerators import DEFAULT, H200, A100, GiB, io_advantage, sram_elements
from arith.kv_cache import BF16, FP8, cache_bytes, concurrency, schemes, latency_floor
from arith.model_d import MODEL_D, total_params

d = MODEL_D
gqa = schemes()["D GQA"]["bytes"]

# ---- the roofline, as the one-line model it is: a kernel of intensity I on a
# machine of peak P and bandwidth W runs at min(P, I W).
def achieved(I, a=DEFAULT):
    return min(a.flops_bf16, I * a.bandwidth)


for I in (1.0, 4.0, 32.0, 100.0):
    assert achieved(I) == I * DEFAULT.bandwidth      # all below the ridge
assert abs(achieved(DEFAULT.balance) / DEFAULT.flops_bf16 - 1) < 1e-12
assert achieved(4000.0) == DEFAULT.flops_bf16       # and saturated above it
# so below the ridge the delivered rate is proportional to the intensity, which
# is another way of saying the time is set by the bytes alone
assert achieved(2.0) / achieved(4.0) == 0.5
print("at I = 4 the part delivers %.1f TFLOP/s of its %.0f: the FLOP count is "
      "not the binding constraint" % (achieved(4.0) / 1e12,
                                      DEFAULT.flops_bf16 / 1e12))
# and the ridge is a property of the machine, not of the model
assert round(A100.balance, 1) == 153.0 and round(H200.balance, 1) == 295.2
assert H200.balance > A100.balance
print("ridge point: %.1f FLOP/byte on the %s, %.1f on the %s"
      % (H200.balance, H200.name, A100.balance, A100.name))

# ---- the second bound of (11.4): how many sequences the scheduler may admit.
# Weights are shared and the cache is not, which is the whole asymmetry.
w = 2 * total_params(d)
assert round(w / GiB, 3) == 14.958
assert round(DEFAULT.capacity_gib, 1) == 131.3      # 141 GB decimal, as sold
long = concurrency(gqa, 131_072, w)
short = concurrency(gqa, 8_192, w)
assert round(long["free_gib"], 2) == round(short["free_gib"], 2) == 116.36
assert round(long["exact"], 2) == 7.27 and long["admit"] == 7
assert round(short["exact"], 2) == 116.36 and short["admit"] == 116
# the floor is not pedantry: admitting the fractional part means preemption
assert long["exact"] - long["admit"] > 0.25
# the two rows are one division at two context lengths, so the exact figures
# are in the ratio 16 while the admitted counts are not
assert abs(short["exact"] / long["exact"] - 16.0) < 1e-9
assert short["admit"] / long["admit"] > 16.0
print("weights %.2f GiB of %.1f GiB, so %.2f GiB free: %d sequences at s = "
      "131072, %d at s = 8192" % (w / GiB, DEFAULT.capacity_gib,
                                  long["free_gib"], long["admit"], short["admit"]))

# ---- and what precision buys, since p_b is the only free factor in (11.2)
# that costs nothing architecturally.
fp8 = schemes(p_b=FP8)["D GQA"]["bytes"]
assert fp8 == 65_536 == gqa // 2
assert abs(concurrency(fp8, 131_072, w)["exact"] / long["exact"] - 2.0) < 1e-12
assert concurrency(fp8, 131_072, w)["admit"] == 14 == 2 * long["admit"]
assert (fp8 * 32_768 * 16) / GiB == 32.0 and fp8 * 32_768 * 16 + w < 80e9
assert latency_floor(cache_bytes(fp8, 131_072)) * 2 == latency_floor(cache_bytes(gqa, 131_072))
print("fp8 halves the cache, doubles the admitted concurrency to %d and halves "
      "the latency floor to %.2f ms"
      % (concurrency(fp8, 131_072, w)["admit"],
         1e3 * latency_floor(cache_bytes(fp8, 131_072))))

# ---- §11.7's constant, for completeness: tiling changes the constant on the
# IO term and not the exponent on s, which is the sentence the section exists
# to make.
assert sram_elements(H200) == 116_736
assert round(io_advantage(d.d_h, H200), 2) == 7.12
assert io_advantage(d.d_h) < 10
print("tiled attention moves M/d_h^2 = %.2fx fewer HBM elements at d_h = %d, "
      "a modest constant and not a change of exponent"
      % (io_advantage(d.d_h), d.d_h))
'''

S3 = r'''
from arith.model_s import MODEL_S, mla_params

SEED = 11003
rng = np.random.default_rng(SEED)

# Reduced widths, because the identity is about associativity and not about
# size; Model S's own numbers appear at the end of the cell, where the ratio
# they produce is the point.
c = MODEL_S
d, d_c, d_h = 256, c.d_c // 8, 32          # a scaled stand-in, same algebra
n_q, n_kv_tokens = 7, 11

X = rng.standard_normal((n_q, d))
C = rng.standard_normal((n_kv_tokens, d_c))          # the cached latents
W_q = rng.standard_normal((d, d_h)) / np.sqrt(d)
W_uk = rng.standard_normal((d_c, d_h)) / np.sqrt(d_c)

# ---- steps 3 to 6.  The naive route materialises k_j; the absorbed route
# never forms it.  Associativity is the entire trick, so it is checked rather
# than asserted.
Q = X @ W_q
K = C @ W_uk                                          # the tensor we want gone
naive = Q @ K.T

W_tilde = W_q @ W_uk.T                                # folded once, at load time
assert W_tilde.shape == (d, d_c)
absorbed = (X @ W_tilde) @ C.T
assert np.abs(naive - absorbed).max() < 1e-10, np.abs(naive - absorbed).max()
print("logits agree to %.2e, and the absorbed route never forms K at all"
      % np.abs(naive - absorbed).max())

# ---- step 8: the same move on the value side, where W_UV folds into W_O and
# the mixing happens on the latents directly.
W_uv = rng.standard_normal((d_c, d_h)) / np.sqrt(d_c)
W_o = rng.standard_normal((d_h, d)) / np.sqrt(d_h)
A = rng.dirichlet(np.ones(n_kv_tokens), size=n_q)     # any row-stochastic mix
naive_out = (A @ (C @ W_uv)) @ W_o
absorbed_out = (A @ C) @ (W_uv @ W_o)
assert (W_uv @ W_o).shape == (d_c, d)
assert np.abs(naive_out - absorbed_out).max() < 1e-10
print("value side: %.2e, and W_UV W_O is a %d x %d constant of the model"
      % (np.abs(naive_out - absorbed_out).max(), d_c, d))

# ---- what the fold costs, in the only currency that matters here.  Per query
# token the naive route reconstructs every cached key; the absorbed route does
# one projection into latent space and then a single inner product per token.
def per_token_flops(n_cached, use_absorption):
    if use_absorption:
        return 2 * d * d_c + 2 * n_cached * d_c       # x W~, then against C
    return 2 * n_cached * d_c * d_h + 2 * d * d_h + 2 * n_cached * d_h


for n_cached in (4096, 32768, 131072):
    naive_f, abs_f = per_token_flops(n_cached, False), per_token_flops(n_cached, True)
    assert abs_f < naive_f
    assert naive_f / abs_f > 5
print("at %d cached tokens the absorbed route is %.1fx cheaper per query token"
      % (131072, per_token_flops(131072, False) / per_token_flops(131072, True)))

# ---- the assumption, and the failure mode.  Absorption needs the two frozen
# factors to be ADJACENT.  Put a per-head normalisation of the reconstructed
# key between them and the identity dies, silently: the shapes still work and
# the numbers no longer agree.
K_normed = K / np.linalg.norm(K, axis=1, keepdims=True)
broken = Q @ K_normed.T
rel = np.linalg.norm(broken - absorbed) / np.linalg.norm(absorbed)
assert broken.shape == absorbed.shape                 # no shape check catches it
assert rel > 0.1, rel
# and a normalisation applied to the CACHED object instead is fine, because it
# acts on c_j itself and not between the two matrices being folded
C_normed = C / np.linalg.norm(C, axis=1, keepdims=True)
assert np.abs((X @ W_tilde) @ C_normed.T - (X @ W_q) @ (C_normed @ W_uk).T).max() < 1e-10
print("normalising the reconstructed key breaks absorption by %.0f%% relative "
      "with no shape error; normalising the latent does not break it at all"
      % (100 * rel))

# ---- and the ratio the section opens with, at Model S's real widths.
assert c.d_c == 512 and c.h * c.d_h == 16384
counterfactual = 2 * c.h * c.d_h                     # K and V, per head
assert counterfactual == 32768
assert c.d_c / counterfactual == 1 / 64
assert counterfactual / c.d_c == 64
assert mla_params(c) > 0
print("Model S: %d latent elements per token per layer against %d for the "
      "same shape with a per-head cache, a factor of %d"
      % (c.d_c, counterfactual, counterfactual // c.d_c))
'''

S4 = r'''
from arith.accelerators import GiB
from arith.kv_cache import BF16, latent_bytes_per_token
from arith.model_s import MODEL_S

SEED = 11004
rng = np.random.default_rng(SEED)
c = MODEL_S
d_h, d_r = 64, 16                       # small stand-ins, the same algebra


def R(delta, width=d_h, base=10000.0):
    """The block-diagonal rotation of §4.3, acting on the right of row vectors."""
    th = base ** (-np.arange(0, width, 2) / width) * delta
    M = np.zeros((width, width))
    for p, (co, si) in enumerate(zip(np.cos(th), np.sin(th))):
        M[2 * p:2 * p + 2, 2 * p:2 * p + 2] = [[co, si], [-si, co]]
    return M


# ---- steps 1 and 2.  R is orthogonal and the product telescopes to R_{i-j},
# which is the whole point of the construction and is cited, not reproved.
for i, j in ((5, 2), (17, 17), (0, 9), (131, 3)):
    assert np.abs(R(i) @ R(i).T - np.eye(d_h)).max() < 1e-12
    assert np.abs(R(i) @ R(j).T - R(i - j)).max() < 1e-12, (i, j)
print("R_i R_j^T = R_{i-j} to %.1e, for every offset tried"
      % max(np.abs(R(i) @ R(j).T - R(i - j)).max()
            for i, j in ((5, 2), (0, 9), (131, 3))))

# ---- step 4: attempt the fold and watch it fail.  R_{i-j} sits BETWEEN the
# two frozen matrices, and no regrouping moves it out.
d, d_c = 128, 48
W_q = rng.standard_normal((d, d_h)) / np.sqrt(d)
W_uk = rng.standard_normal((d_c, d_h)) / np.sqrt(d_c)
folded = {delta: W_q @ R(delta) @ W_uk.T for delta in (0, 1, 7, 129)}
assert all(m.shape == (d, d_c) for m in folded.values())     # shapes are fine
for delta in (1, 7, 129):
    rel = np.linalg.norm(folded[delta] - folded[0]) / np.linalg.norm(folded[0])
    assert rel > 0.1, (delta, rel)
print("the bracket W^Q R_delta W^UK^T is a different %d x %d matrix at every "
      "offset: %s relative to delta = 0"
      % (d, d_c, ["%.2f" % (np.linalg.norm(folded[x] - folded[0])
                            / np.linalg.norm(folded[0])) for x in (1, 7, 129)]))
# associativity is all we had, and it does not help: the obstruction is that
# matrix multiplication does not commute
assert np.abs((W_q @ R(7)) @ W_uk.T - W_q @ (R(7) @ W_uk.T)).max() < 1e-12
# and it is non-commutativity that blocks it: the rotation and the projection
# do not swap, so there is nowhere for R to go
sq = W_q.T @ W_q
assert np.abs(R(7) @ sq - sq @ R(7)).max() > 1e-3

# ---- step 5, and the number that decides it.  Precomputing one folded matrix
# per offset, at Model S's widths and extended context.
elements = c.d * c.d_c * c.extended_context
assert round(elements / 1e11, 2) == 4.81
assert round(elements * BF16 / GiB) == 896
print("one folded %d x %d matrix per offset, over %d offsets: %.2e elements = "
      "%.0f GiB in bf16, per head, per layer"
      % (c.d, c.d_c, c.extended_context, elements, elements * BF16 / GiB))

# ---- step 7, the repair.  A dot product is a sum over coordinates, so
# concatenation splits it additively and the two halves can be treated
# differently.
n_q, n_k = 6, 9
X = rng.standard_normal((n_q, d))
C = rng.standard_normal((n_k, d_c))
W_qr = rng.standard_normal((d, d_r)) / np.sqrt(d)        # rotated query part
W_kr = rng.standard_normal((d, d_r)) / np.sqrt(d)        # rotated key part
Xk = rng.standard_normal((n_k, d))                       # the key tokens' stream

a_i = X @ W_q                                            # any two vectors
b_j = C @ W_uk
u_i = X @ W_qr
v_j = Xk @ W_kr
cat_q = np.concatenate([a_i, u_i], axis=1)
cat_k = np.concatenate([b_j, v_j], axis=1)
assert np.abs(cat_q @ cat_k.T - (a_i @ b_j.T + u_i @ v_j.T)).max() < 1e-12

# the first term absorbs, the second is a d_r-wide rotated product, and the
# sum is the logit.  Written out at three query positions against three keys.
W_tilde = W_q @ W_uk.T
for i in range(n_q):
    for j in range(n_k):
        rot = (u_i[i] @ R(i, d_r)) @ (v_j[j] @ R(j, d_r))
        relative = u_i[i] @ R(i - j, d_r) @ v_j[j]
        assert abs(rot - relative) < 1e-10, (i, j)
        logit = (X[i] @ W_tilde) @ C[j] + rot
        naive = (X[i] @ W_q) @ (C[j] @ W_uk) + rot
        assert abs(logit - naive) < 1e-10
print("the split logit equals the unsplit one to %.1e, the absorbing half "
      "folds, and the rotated half is relative by step 2" % 1e-10)

# ---- step 8, the cache accounting, and the asymmetry in it.  The rotated KEY
# part is shared across heads; the rotated QUERY part is not, because each head
# needs its own view of position while position itself is common.
assert c.d_c + c.d_r == 576
assert latent_bytes_per_token(c.L, c.d_c, c.d_r, BF16) == 70_272
per_head_if_not_shared = c.d_c + c.h * c.d_r
assert per_head_if_not_shared == 512 + 128 * 64 == 8704
assert round(per_head_if_not_shared / (c.d_c + c.d_r), 1) == 15.1
print("cache is d_c + d_r = %d elements per token per layer (%d bytes over %d "
      "layers); giving every head its own rotated key part would make it %d, "
      "%.1fx more" % (c.d_c + c.d_r,
                      latent_bytes_per_token(c.L, c.d_c, c.d_r, BF16), c.L,
                      per_head_if_not_shared,
                      per_head_if_not_shared / (c.d_c + c.d_r)))

# ---- the failure mode: applying RoPE to the compressed part as well, for
# consistency, reinstates the obstruction and silently costs the absorption.
rotated_compressed = {delta: W_q @ R(delta) @ W_uk.T for delta in (0, 5)}
assert np.linalg.norm(rotated_compressed[5] - rotated_compressed[0]) > 0.1
print("rotating the compressed part too puts R_{i-j} back between the two "
      "frozen factors, and absorption is gone again")
'''

S5 = r'''
from fractions import Fraction

SEED = 11005
rng = np.random.default_rng(SEED)

# ---- the algorithm, written once, with the partition, the visit order and the
# dtype all left free, because the corollary is that none of them matters.
def online(z, v, order, n_blocks, dtype=np.float64):
    z = np.asarray(z, dtype=dtype)
    v = np.asarray(v, dtype=dtype)
    m = dtype(-1e4 if dtype == np.float16 else -1e30)
    ell = dtype(0.0)
    o = np.zeros(v.shape[1], dtype=dtype)
    for B in np.array_split(order, n_blocks):
        if len(B) == 0:
            continue
        m_new = max(m, z[B].max())
        r = np.exp(m - m_new)                 # the rescaling factor of (11.20)
        w = np.exp(z[B] - m_new)
        ell = r * ell + w.sum()
        o = r * o + (w[:, None] * v[B]).sum(0)
        m = m_new
    return (o / ell).astype(np.float64)


# ---- FIRST, the exactness, and it has to be demonstrated in an arithmetic
# that is exact.  Replace e^z by 2^z on integer logits: softmax is the same
# object with a different base, every weight is then a dyadic rational, and the
# recurrence can be run over the rationals with no rounding anywhere.
n, w_dim = 40, 3
zi = rng.integers(-30, 30, n)
vi = rng.integers(-9, 9, (n, w_dim))


def online_exact(order, n_blocks):
    m = None
    ell = Fraction(0)
    o = [Fraction(0)] * w_dim
    for B in np.array_split(np.asarray(order), n_blocks):
        if len(B) == 0:
            continue
        blk_max = int(max(zi[j] for j in B))
        m_new = blk_max if m is None else max(m, blk_max)
        r = Fraction(2) ** (0 if m is None else int(m) - m_new)
        ell = r * ell + sum(Fraction(2) ** (int(zi[j]) - m_new) for j in B)
        o = [r * o[t] + sum(Fraction(2) ** (int(zi[j]) - m_new) * int(vi[j, t])
                            for j in B) for t in range(w_dim)]
        m = m_new
    return [x / ell for x in o]


mx = int(zi.max())
den = sum(Fraction(2) ** (int(zz) - mx) for zz in zi)
reference = [sum(Fraction(2) ** (int(zi[j]) - mx) * int(vi[j, t])
                 for j in range(n)) / den for t in range(w_dim)]
for n_blocks in (1, 2, 3, 7, 13, n):
    assert online_exact(np.arange(n), n_blocks) == reference, n_blocks
assert online_exact(rng.permutation(n), 7) == reference          # any visit order
assert online_exact(np.arange(n)[::-1], 5) == reference
print("over the rationals the recurrence is EQUAL to the unblocked reference, "
      "for every partition and every visit order: %s" % [str(x)[:14] for x in reference])

# ---- SECOND, floating point, where the same statement is false as an
# equality.  The rescalings introduce O(T u) relative rounding, so the right
# assertion is closeness with a tolerance, and asserting equality would be
# asserting the wrong thing about a correct algorithm.
s, d_h = 4096, 128
z = rng.normal(scale=3.0, size=s)
v = rng.normal(size=(s, d_h))
idx = np.arange(s)

worst = {}
for dtype, name, tol in ((np.float64, "fp64", 1e-13),
                         (np.float32, "fp32", 1e-5),
                         (np.float16, "fp16", 3e-2)):
    zz, vv = z.astype(dtype), v.astype(dtype)
    w = np.exp(zz - zz.max())
    ref = ((w[:, None] * vv).sum(0) / w.sum()).astype(np.float64)
    errs = [np.abs(online(z, v, idx, nb, dtype) - ref).max()
            for nb in (2, 3, 7, 64, 512)]
    worst[name] = max(errs)
    assert worst[name] < tol, (name, worst[name])
    # and never bitwise equal, in any of the three
    for nb in (2, 7, 64, 512):
        assert not np.array_equal(online(z, v, idx, nb, dtype), ref), (name, nb)
assert worst["fp64"] < worst["fp32"] < worst["fp16"]
print("floating point, s = %d, partitions from 2 blocks to 512: %s in fp64, "
      "%s in fp32, %s in fp16, and never bitwise equal in any of them"
      % (s, "%.1e" % worst["fp64"], "%.1e" % worst["fp32"], "%.1e" % worst["fp16"]))

# ---- the corollary, which is what licenses a scheduler to pick tiles on
# hardware grounds alone: ragged partitions and out-of-order visits land in the
# same place, to the same rounding.
ref64 = ((np.exp(z - z.max())[:, None] * v).sum(0) / np.exp(z - z.max()).sum())
for nb in (1, 2, 3, 5, 7, 11, 64, 512, 1000, s):
    assert np.abs(online(z, v, idx, nb) - ref64).max() < 1e-13, nb
for _ in range(20):
    perm = rng.permutation(s)
    assert np.abs(online(z, v, perm, 37) - ref64).max() < 1e-13
print("ten block counts and twenty random visit orders all agree with the "
      "unblocked reference to better than 1e-13")

# ---- the implementation trap the assumptions clause names.  A causal mask is
# applied by setting z = -inf BEFORE the recurrence, and e^{-inf - m} must
# evaluate to zero rather than to a NaN, which is why m is initialised to a
# large finite negative number.
z_masked = z.copy()
z_masked[s // 2:] = -np.inf
with np.errstate(invalid="ignore"):
    got = online(z_masked, v, idx, 64)
keep = np.isfinite(z_masked)
wk = np.exp(z_masked[keep] - z_masked[keep].max())
want = (wk[:, None] * v[keep]).sum(0) / wk.sum()
assert np.isfinite(got).all(), "a NaN here is the -inf minus -inf trap"
assert np.abs(got - want).max() < 1e-13
# whereas starting from m = -inf produces exactly that NaN on the first block
assert np.isnan(np.exp(-np.inf - (-np.inf)))
print("masked rows contribute nothing and the result is finite; initialising "
      "m to -inf instead gives exp(-inf + inf), which is a NaN")
'''

SECTIONS = [
    ("1", "The KV cache, and the arithmetic intensity of decode",
     "The leading two in the cache formula is K-and-V, two tensors, and it is "
     "not bytes per element, which is a separate factor that also happens to "
     "equal two in bf16. The cell keeps them apart, sizes the error of "
     "collapsing them, and then divides the decode FLOP count by the decode "
     "byte count over a grid of shapes to confirm that the depth, the head "
     "width and the context length all cancel. What survives is the ratio of "
     "query heads to key/value heads and the precision of the cache, which for "
     "Model D is four FLOPs per byte against a machine balance of nearly three "
     "hundred.",
     S1),
    ("2", "Bandwidth, not FLOPs",
     "Below the ridge a kernel runs at its intensity times the bandwidth and "
     "its FLOP count is irrelevant, so the two numbers a deployment is sized "
     "by are a latency floor and an admitted concurrency, both of them "
     "divisions of the cache formula. The cell computes both at Model D's "
     "shape on a 141 GB part, checks that the free memory is the same whatever "
     "the context length while the number of sequences it buys is not, and "
     "shows what halving the cache precision buys in each.",
     S2),
    ("3", "The absorption identity",
     "Reconstruction from a latent looks like work the cache existed to avoid, "
     "and one line of algebra shows it is not: the query projection and the "
     "key up-projection are adjacent frozen matrices, so they fold into one "
     "constant of the model and the key is never formed. The cell checks the "
     "identity on both the score side and the value side, prices the saving, "
     "and then breaks it deliberately by putting a per-head normalisation "
     "between the two folded factors, which is the failure mode the "
     "derivation names and which no shape check can catch.",
     S3),
    ("4", "Why the rotation cannot be absorbed, and the decoupled repair",
     "The rotation telescopes to a function of the offset, and that function "
     "sits between the two matrices we wanted to fold. Associativity is all the "
     "previous section had and it does not move a factor past another one, so "
     "the fold fails, and the cell measures how far the bracket moves with the "
     "offset. Precomputing one folded matrix per offset costs 896 GiB per head "
     "per layer at Model S's widths, so the repair is to split each head into "
     "a compressed part that never rotates and a narrow part that carries all "
     "the position, with the rotated key part shared across heads and the "
     "rotated query part not.",
     S4),
    ("5", "Online softmax is exact",
     "Exact is a strong word and it is meant literally, so the exactness is "
     "demonstrated in an arithmetic that is exact: base-two softmax on integer "
     "logits makes every weight a dyadic rational, and the recurrence run over "
     "the rationals is equal to the unblocked reference for every partition "
     "and every visit order. In floating point the same algorithm is not "
     "bitwise equal to that reference and never will be, because the rescalings "
     "round, so the cell asserts closeness with a tolerance in three precisions "
     "and asserts inequality of the bits. An exact algorithm in an inexact "
     "arithmetic is a different claim from an approximate algorithm, and a "
     "naive equality assertion would be testing the wrong one.",
     S5),
]
