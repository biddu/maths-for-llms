"""The CI guard.  Every number printed in an arithmetic box is recomputed here,
so the print edition and the repository cannot drift apart."""
from arith.model_d import (MODEL_D, non_embedding, embedding, total_params,
                           per_layer, attention_params, attention_flops, crossover,
                           rope_bands, critical_dimension, norm_stats,
                           llama_intermediate_size, ffn_budget, mlp_flops,
                           activation_memory_backward, checkpoint_memory,
                           scaling_loss, loss_units, tokens_for_loss,
                           label_smoothing, REFIT_2024, CHINCHILLA,
                           BYTES_PER_TOKEN, TRAINED_TOKENS,
                           params_by_ndim, optimiser_state,
                           adam_burst_bound, beta2_half_life,
                           newton_schulz_poly)
from arith.small_model import SMALL
from arith import model_s


def test_model_d_ledger():
    assert per_layer(MODEL_D)["total"] == 218_112_000
    assert non_embedding(MODEL_D) == 6_979_588_096
    assert embedding(MODEL_D) == 525_336_576
    assert total_params(MODEL_D) == 8_030_261_248


def test_chapter2_tying_box():
    assert 2 * embedding(MODEL_D) == 1_050_673_152
    assert non_embedding(MODEL_D) + embedding(MODEL_D) == 7_504_924_672
    assert round(200 * embedding(MODEL_D) / total_params(MODEL_D), 2) == 13.08
    assert non_embedding(SMALL) == 973_146_112
    assert non_embedding(SMALL) + 2 * embedding(SMALL) == 1_498_482_688
    assert non_embedding(SMALL) + embedding(SMALL) == 1_235_814_400


def test_chapter3_attention_box():
    p = attention_params(MODEL_D)
    assert p["W_Q"] == 16_777_216 and p["W_K"] == 4_194_304
    assert sum(p.values()) == 41_943_040
    f = attention_flops(MODEL_D, 8192)
    assert round(f["projections"] / 1e9, 2) == 687.19
    assert round(f["attention"] / 2 / 1e9, 2) == 549.76
    assert crossover(MODEL_D) == 5120 == int(1.25 * MODEL_D.d)
    assert 2 * MODEL_D.h * MODEL_D.d_h == 8192          # MHA counterfactual


def test_model_s_expert_arithmetic():
    t = model_s.totals()
    assert t["expert"] == 44_040_192
    assert t["moe_layer_active"] == 396_361_728
    assert round(t["total"] / 1e9) == 670        # see NOTE in model_s.py
    assert round(t["active"] / 1e9) == 37


def test_chapter4_rope_box():
    import math
    rows = rope_bands(MODEL_D)
    assert critical_dimension(MODEL_D) == 35
    assert round(rows[0]["lambda"], 2) == 6.28
    assert round(rows[34]["lambda"]) == 6695 and round(rows[35]["lambda"]) == 8219
    assert sum(r["gamma"] >= 1 - 1e-12 for r in rows) == 19       # extrapolate
    assert sum(r["gamma"] <= 1e-12 for r in rows) == 29           # interpolate
    assert round(rows[19]["effective_scale"], 2) == 1.20
    assert round(rows[34]["effective_scale"], 2) == 14.44
    assert round(rows[35]["effective_scale"], 2) == 16.00
    t = 0.1 * math.log(16) + 1
    assert round(t, 4) == 1.2773 and round(t * t, 4) == 1.6314
    assert round(MODEL_D.rope_base * 16 ** (128 / 126) / 1e6, 2) == 8.36


def test_chapter5_norm_box():
    import math
    s = norm_stats(MODEL_D)
    assert s["sites"] == 65
    assert s["rmsnorm_params"] == 266_240
    assert s["layernorm_params"] == 532_480
    assert s["qk_norm_params"] == 8_192
    n = non_embedding(MODEL_D)
    assert round(100 * s["rmsnorm_params"] / n, 4) == 0.0038
    assert round(math.sqrt(MODEL_D.d_h), 2) == 11.31
    assert round(math.exp(2 * math.sqrt(MODEL_D.d_h)) / 1e9, 2) == 6.71


def test_chapter6_ffn_box():
    c = MODEL_D
    b = ffn_budget(c)

    # steps 1 and 2: the identity, which is permanent
    assert b["ungated_4d_params"] == 134_217_728 == 2 * c.d * (4 * c.d)
    assert b["d_ff_two_thirds"] == 10_922
    assert b["gated_two_thirds_params"] == 134_209_536
    gap = b["ungated_4d_params"] - b["gated_two_thirds_params"]
    assert gap == 8_192 == 2 * c.d, "the residue is integer rounding and nothing else"
    assert round(100 * gap / b["ungated_4d_params"], 3) == 0.006

    # steps 3 and 4: the recipe, which is a Llama-family choice and is daggered
    p = llama_intermediate_size(c.d)
    assert (p["4d"], p["two_thirds"], p["multiplied"]) == (16_384, 10_922, 14_198)
    assert p["intermediate_size"] == 14_336 == c.d_ff
    assert round(p["ratio_to_two_thirds"], 4) == 1.3125
    assert llama_intermediate_size(8_192)["intermediate_size"] == 28_672
    assert 14_336 % 1024 == 0 and 14_336 % 256 == 0
    assert 14_336 % 8 == 0 and 10_922 % 8 != 0        # eight-way tensor parallel

    # step 5: where the parameters are
    assert b["mlp_per_layer"] == 176_160_768
    assert b["mlp_total"] == 5_637_144_576
    assert b["attn_total"] == 1_342_177_280
    assert b["norm_total"] == 266_240 == norm_stats(c)["rmsnorm_params"]
    assert b["non_embedding"] == non_embedding(c) == 6_979_588_096
    assert round(100 * b["ffn_share"], 1) == 80.8
    assert b["non_embedding_two_thirds"] == 5_637_148_672
    assert b["saving"] == 1_342_439_424
    assert round(100 * b["saving_frac"], 1) == 19.2
    # the near-collision the author note warns a proof-reader about
    assert b["non_embedding_two_thirds"] - b["mlp_total"] == c.d

    # activation memory, and the FLOP crossover of equation (6.17)
    assert round(2 * b["d_ff_two_thirds"] / (4 * c.d), 4) == 1.3333
    ffn = mlp_flops(c, 1)
    proj = attention_flops(c, 1)["projections"]
    assert round(ffn / 1e6, 1) == 352.3 and round(proj / 1e6, 1) == 83.9
    assert (ffn - proj) / (4 * c.d) == 16_384 == 4 * c.d
    assert round((proj + 4 * 8_192 * c.d) / ffn, 3) == 0.619
    assert round((proj + 4 * 131_072 * c.d) / ffn, 3) == 6.333


def test_chapter6_gelu_constants():
    import math
    from scipy.optimize import brentq
    from scipy.stats import norm
    dgelu = lambda x: norm.cdf(x) + x * norm.pdf(x)
    xstar = brentq(dgelu, -3.0, -0.1, xtol=1e-14)
    assert round(xstar, 4) == -0.7518
    assert round(xstar * norm.cdf(xstar), 4) == -0.1700
    assert round(2 * norm.pdf(0), 4) == 0.7979 == round(math.sqrt(2 / math.pi), 4)
    assert round(1 / math.sqrt(2 * math.pi), 4) == 0.3989

    import numpy as np
    tanh_form = lambda x: 0.5 * x * (1 + np.tanh(
        math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
    xs = np.linspace(-8, 8, 200_001)
    err = np.abs(xs * norm.cdf(xs) - tanh_form(xs))
    assert round(err.max() * 1e4, 2) == 4.73
    assert round(xs[err.argmax()], 3) == 2.699
    assert round(2.0 ** -8 * 1e3, 2) == 3.91          # one bf16 ulp near 1


def test_chapter6_gating_algebra():
    """D-6.3, checked rather than asserted: the polarisation identity, the fact
    that only the symmetric part of a quadratic form survives, and the rank-2
    eigen-pairing of equation (6.13)."""
    import numpy as np
    rng = np.random.default_rng(0)
    d = 7
    w, v = rng.normal(size=d), rng.normal(size=d)
    X = rng.normal(size=(500, d))
    lhs = (X @ w) * (X @ v)
    rhs = 0.25 * ((X @ (w + v)) ** 2 - (X @ (w - v)) ** 2)
    assert np.abs(lhs - rhs).max() < 1e-12

    M = np.outer(w, v)
    assert np.linalg.matrix_rank(M) == 1 and not np.allclose(M, M.T)
    q = np.einsum("ij,jk,ik->i", X, M, X)
    qs = np.einsum("ij,jk,ik->i", X, (M + M.T) / 2, X)
    assert np.abs(q - qs).max() < 1e-12

    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    a, b = 1.0, 1.0
    wp, vp = a * Q[:, 0] + b * Q[:, 5], a * Q[:, 0] - b * Q[:, 5]
    H = 0.5 * (np.outer(wp, vp) + np.outer(vp, wp))
    ev = np.sort(np.linalg.eigvalsh(H))
    assert np.linalg.matrix_rank(H) == 2
    assert round(ev[0], 10) == -b ** 2 and round(ev[-1], 10) == a ** 2


def test_chapter7_backward_box():
    c = MODEL_D
    a = activation_memory_backward(c, b=1, s=8192, dtype="bf16")
    it = a["items"]
    assert it["x, x_hat1, Q, O_cat, y, x_hat2"] == 402_653_184
    assert it["K, V"] == 33_554_432
    assert it["P = softmax(S)"] == 4_294_967_296
    assert it["G, U, A"] == 704_643_072
    assert it["r1, r2 (fp32)"] == 65_536
    assert a["total"] == 5_435_883_520
    assert round(a["total"] / 2**30, 2) == 5.06
    assert round(100 * a["probs_share"], 1) == 79.0
    assert round(a["per_stack"] / 1e9, 1) == 173.9

    f = activation_memory_backward(c, b=1, s=8192, dtype="bf16", fused_attention=True)
    assert round(f["total"] / 1e9, 3) == 1.141
    assert round(f["per_stack"] / 1e9, 1) == 36.5

    # halving the batch changes no ratio; halving the sequence quarters P alone
    half_b = activation_memory_backward(c, b=1, s=8192)["total"]
    assert activation_memory_backward(c, b=2, s=8192)["total"] == 2 * half_b
    assert (activation_memory_backward(c, b=1, s=4096)["items"]["P = softmax(S)"]
            == it["P = softmax(S)"] // 4)


def test_chapter7_checkpointing():
    r = checkpoint_memory(MODEL_D, s=8192)
    assert round(r["M_b"] / 1e6, 1) == 67.1
    assert round(r["ratio"], 1) == 81.0 and r["clipped"] is True
    assert round(r["m_star"], 1) == 50.9 and r["m"] == 32
    assert round(r["M"] / 1e9, 2) == 7.58

    f = checkpoint_memory(MODEL_D, s=8192, fused_attention=True)
    assert round(f["ratio"], 1) == 17.0 and f["clipped"] is False
    assert round(f["m_star"], 1) == 23.3
    assert round(f["M"] / 1e9, 2) == 3.13 == round(f["M_sqrt_rule"] / 1e9, 2)
    assert round(f["unchecked"] / 1e9, 1) == 36.5

    # E-7.6: at s = 2048 the stored-P case is still clipped, but only just
    e = checkpoint_memory(MODEL_D, s=2048)
    assert round(e["ratio"], 1) == 33.0 and e["clipped"] is True
    assert round(e["m_star"], 2) == 32.50 and round(e["M"] / 1e9, 3) == 1.091
    # and with P recomputed the ratio is sequence-independent
    for s in (1024, 2048, 8192, 32768):
        assert round(checkpoint_memory(MODEL_D, s=s, fused_attention=True)["ratio"], 1) == 17.0


def test_chapter7_gradient_flow_bound():
    """D-7.4, equation (7.28)."""
    import math
    L = MODEL_D.L
    assert round((1 + 1 / L) ** L - 1, 3) == 1.677
    assert round(math.e - 1, 3) == 1.718
    assert round(1.1 ** L - 1, 1) == 20.1
    # the bound is O(1) exactly when delta = O(1/L): check it stays bounded
    for LL in (8, 32, 128, 1024):
        assert (1 + 1 / LL) ** LL - 1 < math.e - 1
    # and explodes when delta does not shrink with L
    assert 1.1 ** 128 - 1 > 1e5


def test_chapter7_activation_and_precision_constants():
    """The SiLU derivative facts of (7.11), and Table 7.2."""
    import math
    from scipy.optimize import brentq, minimize_scalar
    sig = lambda z: 1 / (1 + math.exp(-z))
    dsilu = lambda g: sig(g) * (1 + g * (1 - sig(g)))
    assert abs(dsilu(0.0) - 0.5) < 1e-15
    root = brentq(dsilu, -5, -0.5, xtol=1e-14)
    assert round(root, 4) == -1.2785
    lo = minimize_scalar(dsilu, bounds=(-6, 0), method="bounded",
                         options={"xatol": 1e-12})
    hi = minimize_scalar(lambda g: -dsilu(g), bounds=(0, 6), method="bounded",
                         options={"xatol": 1e-12})
    assert round(lo.fun, 4) == -0.0998 and round(-hi.fun, 4) == 1.0998
    assert round(lo.x, 4) == -round(hi.x, 4) == -2.3994
    # SiLU'(g) + SiLU'(-g) = 1, which is why the extrema mirror about 1/2
    for g in (-4.0, -1.0, 0.3, 2.7, 9.0):
        assert abs(dsilu(g) + dsilu(-g) - 1.0) < 1e-14

    for bits, u in ((23, 5.96e-08), (10, 4.88e-04), (7, 3.91e-03)):
        assert abs(2.0 ** -(bits + 1) / u - 1) < 2e-3


def test_chapter8_loss_box():
    """The loss is derived from Chapter 10's refit, not transcribed from a log,
    so if any of the five coefficients moves this test catches it."""
    f = REFIT_2024
    assert (f["L_inf"], f["A"], f["alpha"], f["B"], f["beta"]) == \
        (1.82, 482.0, 0.348, 2085.4, 0.366)
    N, D = non_embedding(MODEL_D), TRAINED_TOKENS
    s = scaling_loss(N, D)
    assert round(s["parameter_term"], 3) == 0.181
    assert round(s["data_term"], 3) == 0.031
    assert round(s["L"], 4) == 2.0323 and round(s["L"], 2) == 2.03
    assert round(s["floor_at_fixed_N"], 4) == 2.0009
    # the headline: the data term is a small fraction of the printed loss
    assert round(100 * s["data_term"] / s["L"], 1) == 1.5


def test_chapter8_units():
    u = loss_units(2.03, BYTES_PER_TOKEN)
    assert round(u["bits_per_token"], 3) == 2.929
    assert round(u["perplexity"], 3) == 7.614
    assert round(u["bits_per_byte"], 3) == 0.771
    assert round(u["uniform_nats"], 3) == 11.762
    assert u["uniform_ppl"] == MODEL_D.V
    assert round(100 * u["fraction_of_uniform"]) == 17
    # PPL = exp(CE_nats) = 2^CE_bits is one identity, not two measurements
    assert abs(u["perplexity"] - 2 ** u["bits_per_token"]) < 1e-9
    v = loss_units(1.925, BYTES_PER_TOKEN)
    assert round(v["perplexity"], 3) == 6.855
    assert round(v["bits_per_token"], 3) == 2.777
    assert round(v["bits_per_byte"], 3) == 0.731


def test_chapter8_cost_of_a_tenth():
    assert round(1 / REFIT_2024["beta"], 3) == 2.732
    for target, mult, toks in ((1.93, 5.85, 87.8), (1.925, 6.64, 99.7)):
        r = tokens_for_loss(target)
        assert round(r["multiplier"], 2) == mult
        assert round(r["tokens"] / 1e12, 1) == toks
    prev = tokens_for_loss(2.03, L0=2.135)
    assert round(prev["multiplier"], 2) == 3.03
    assert round(prev["tokens"] / 1e12, 1) == 45.4
    # halving the reducible loss costs 2^(1/beta) whatever L_inf is
    assert abs(tokens_for_loss(1.925)["multiplier"] - 2 ** (1 / 0.366)) < 1e-9
    # and no quantity of data reaches the floor
    import pytest
    with pytest.raises(ValueError):
        tokens_for_loss(REFIT_2024["L_inf"])


def test_chapter8_e87_sensitivity():
    """E-8.7: which of the two coefficients moves the answer more?  The floor,
    and by more than the exponent, and in the opposite direction."""
    base = tokens_for_loss(1.93)["tokens"] / 1e12
    exp_only = tokens_for_loss(1.93, beta=CHINCHILLA["beta"])["tokens"] / 1e12
    flr_only = tokens_for_loss(1.93, L_inf=CHINCHILLA["L_inf"])["tokens"] / 1e12
    both = tokens_for_loss(1.93, beta=CHINCHILLA["beta"],
                           L_inf=CHINCHILLA["L_inf"])["tokens"] / 1e12
    assert round(base, 1) == 87.8
    assert round(exp_only, 1) == 151.0
    assert round(flr_only, 1) == 40.2
    assert round(both, 1) == 54.5
    assert (flr_only / base) ** -1 > exp_only / base, \
        "the floor moves it by more than the exponent does"


def test_chapter8_label_smoothing():
    ls = label_smoothing(0.1, MODEL_D.V)
    assert round(ls["ratio"]) == 1_154_305
    assert round(ls["gap"], 4) == 13.9590
    assert round(ls["floor"], 4) == 1.5012
    # the large-V simplification of D-8.3 step 6 is good to six decimals here
    assert abs(ls["gap"] - ls["gap_large_V"]) < 1e-5
    # the 2017 translation setting, for the failure mode
    old = label_smoothing(0.1, 32000)
    assert round(old["floor"], 4) == 1.3624
    assert round(old["gap"], 4) == 12.5707
    # E-8.4: the corrected small-eps form, and where the crude one fails
    import math
    small = label_smoothing(0.01, MODEL_D.V)
    assert abs(small["floor"] - small["floor_small_eps"]) < 1e-4
    assert abs(small["floor"] - 0.01 * math.log(MODEL_D.V)) > 0.05
    from scipy.optimize import brentq
    e1 = brentq(lambda e: label_smoothing(e, MODEL_D.V)["floor"] - 1.0, 1e-6, 0.9)
    assert round(e1, 4) == 0.0647


def test_chapter9_parameter_split():
    p = params_by_ndim(MODEL_D)
    assert p["two_d"] == 6_979_321_856
    assert p["other"] == 1_050_939_392
    assert p["total"] == total_params(MODEL_D) == 8_030_261_248
    # only the 2-D matrices are Muon-eligible; D-9.4's failure mode says why
    assert p["two_d"] + p["embeddings"] + p["gains"] == p["total"]
    assert p["gains"] == norm_stats(MODEL_D)["rmsnorm_params"]


def test_chapter9_optimiser_box():
    a = optimiser_state(MODEL_D, "adamw")
    assert a["bytes_per_param"] == 16 and a["state_per_param"] == 12
    assert round(a["total"] / 1e9, 2) == 128.48
    assert round(a["state"] / 1e9, 2) == 96.36
    for k in ("bf16 parameters", "bf16 gradients"):
        assert round(a["items"][k] / 1e9, 2) == 16.06
    for k in ("fp32 master weights", "first moment m", "second moment v"):
        assert round(a["items"][k] / 1e9, 2) == 32.12

    m = optimiser_state(MODEL_D, "muon")
    assert round(m["items"]["Muon momentum (2-D)"] / 1e9, 2) == 27.92
    assert round(m["items"]["AdamW m and v (rest)"] / 1e9, 2) == 8.41
    assert round(m["total"] / 1e9, 2) == 100.57
    assert round(m["state"] / 1e9, 2) == 68.45
    assert round(m["state_per_param"], 2) == 8.52
    assert round(100 * (1 - m["state"] / a["state"])) == 29
    assert round((a["state"] - m["state"]) / 1e9, 1) == 27.9

    # sharded over eight devices, and the 8-bit variant that still does not fit
    assert round(a["total"] / 8 / 1e9, 2) == 16.06
    assert round(m["total"] / 8 / 1e9, 2) == 12.57
    q = optimiser_state(MODEL_D, "adamw", moment_bytes=1)
    assert round(q["total"] / 1e9, 2) == 80.30 and q["total"] / 1e9 > 80


def test_chapter9_burst_bound_and_half_life():
    import math
    assert round(adam_burst_bound(0.9, 0.999), 3) == 3.162
    assert round(adam_burst_bound(0.9, 0.95), 3) == 0.447
    assert round(adam_burst_bound(0.9, 0.99), 3) == 1.000
    assert adam_burst_bound(0.9, 0.95) < 1.0 < adam_burst_bound(0.9, 0.999)
    assert round(beta2_half_life(0.999)) == 693
    assert round(beta2_half_life(0.95), 1) == 13.5
    assert round(beta2_half_life(0.99), 1) == 69.0
    # the bound is what simulated Adam actually does
    def sim(b1, b2, quiet=8000):
        m = v = 0.0; peak = 0.0
        for t in range(1, quiet + 50):
            g = 1.0 if t == quiet else 1e-6
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            if t >= quiet:
                peak = max(peak, abs((m / (1 - b1 ** t)) / math.sqrt(v / (1 - b2 ** t))))
        return peak
    for b2 in (0.95, 0.99, 0.999):
        assert abs(sim(0.9, b2) - adam_burst_bound(0.9, b2)) < 1e-3


def test_chapter9_newton_schulz_polynomial():
    """D-9.4 step 7.  The polynomial is not converging and its fixed points say so."""
    import numpy as np
    a, b, c = 3.4445, -4.7750, 2.0315
    assert newton_schulz_poly(0.0) == 0.0
    # p'(0) = a, which is why the smallest singular values move fastest
    h = 1e-7
    assert abs((newton_schulz_poly(h) - newton_schulz_poly(-h)) / (2 * h) - a) < 1e-4
    # the two positive fixed points
    r = np.roots([c, 0, b, 0, a - 1, 0])
    fp = sorted(float(x.real) for x in r if abs(x.imag) < 1e-9 and x.real > 1e-6)
    assert [round(x, 3) for x in fp] == [0.868, 1.264]
    # five steps carry everything above 0.0016 into a band around 1.  The
    # lower edge is set by the polynomial's own oscillation and not by the
    # smallest input: it sits at 0.682 whatever the starting threshold.
    s = np.linspace(0.0016, 1.0, 5000)
    out = newton_schulz_poly(s, steps=5)
    assert out.min() >= 0.68 and out.max() <= 1.2025
    assert newton_schulz_poly(np.linspace(0.01, 1.0, 5000), steps=5).min() >= 0.68
    # and it does not converge: seven steps do not shrink the band
    out7 = newton_schulz_poly(s, steps=7)
    assert out7.max() - out7.min() > 0.3


def test_chapter10_arithmetic_box():
    """§10.8, every printed figure.  The two ratios are one number: at fixed N
    the FLOP ratio is the token ratio, and rounding C to two significant
    figures is what made the blueprint's 104.0 and 104.2 disagree."""
    from arith.scaling_budget import box
    b = box()
    assert b["N"] == non_embedding(MODEL_D)
    assert round(b["D_opt"] / 1e9) == 144
    assert round(b["tokens_per_param_opt"], 2) == 20.60
    assert "%.5e" % b["C_opt"] == "6.02037e+21"
    assert round(b["tokens_per_param_ship"], 1) == 2149.1
    assert "%.5e" % b["C_ship"] == "6.28163e+23"
    assert abs(b["token_ratio"] - b["flop_ratio"]) < 1e-9
    assert round(b["token_ratio"], 2) == 104.34
    assert round(b["L_ship"], 5) == 2.03227
    # the equal-loss compute-optimal counterpart, and the break-even
    assert "%.4e" % b["N_c"] == "3.0068e+10"
    assert round(b["D_c"] / 1e9) == 576
    assert "%.4e" % b["extra_train_flops"] == "5.2418e+23"
    assert "%.4e" % b["saved_per_token"] == "4.6177e+10"
    assert "%.4e" % b["break_even_tokens"] == "1.1352e+13"
    assert round(b["break_even_tokens"] / 3e10) == 378


def test_chapter10_flop_accounting():
    """D-10.1's corrections.  6ND is not the training cost and the gap is a
    function of s, which is why a FLOP claim without a context length is not a
    claim.  Model D's block is larger than the d_ff = 8d/3 idealisation, so its
    attention ratio is 0.308 and not the textbook s/(6d) = 0.333."""
    from arith.scaling_budget import true_training_flops
    N = non_embedding(MODEL_D)
    t = true_training_flops(N, TRAINED_TOKENS, 8192)
    assert round(t["attn_ratio"], 3) == 0.308
    assert round(8192 / (6 * MODEL_D.d), 3) == 0.333
    assert round(t["logit_ratio"], 3) == 0.075
    assert round(100 * t["understatement"], 1) == 27.7
    long = true_training_flops(N, TRAINED_TOKENS, 131072)
    assert round(long["attn_ratio"], 2) == 4.92
    assert round(100 * long["understatement"]) == 83
    # and the naive count is the parameter term alone
    assert t["naive_6nd"] == 6 * N * TRAINED_TOKENS


def test_chapter10_allocation_and_drift():
    """D-10.2.  Twenty tokens per parameter is not scale-invariant unless the
    two exponents are equal exactly, and neither fit has them equal."""
    from arith.scaling_budget import (optimal_D, optimal_N,
                                      tokens_per_param_exponent, loss)
    e = tokens_per_param_exponent(REFIT_2024)
    assert round(e["in_N"], 4) == -0.0492
    assert round(100 * e["per_decade_of_N"], 1) == -10.7
    assert round(optimal_D(7e9) / 7e9, 1) == 20.6
    assert round(optimal_D(7e10) / 7e10, 1) == 18.4
    # the published pair drifts the other way, and hard
    ec = tokens_per_param_exponent(CHINCHILLA)
    assert ec["in_N"] > 0 and round(100 * ec["per_decade_of_N"]) == 64
    # N* from the closed form is the minimiser of the isoFLOP curve
    for C in (1e21, 1e23):
        n = optimal_N(C)
        here = loss(n, C / (6 * n))
        for f in (0.9, 1.1):
            assert loss(n * f, C / (6 * n * f)) > here


def test_chapter10_inference_aware():
    """D-10.3.  Serving moves the optimum toward smaller models and more
    tokens, monotonically, and collapses to D-10.2 when nothing is served."""
    from arith.scaling_budget import inference_aware_optimum, optimal_D
    base = inference_aware_optimum(2.20, 0.0)
    assert abs(base["D"] / optimal_D(base["N"]) - 1) < 1e-4
    prev = base
    for d_inf in (1e12, 1e13, 1e14):
        cur = inference_aware_optimum(2.20, d_inf)
        assert cur["N"] < prev["N"]
        assert cur["tokens_per_param"] > prev["tokens_per_param"]
        prev = cur
    assert prev["tokens_per_param"] > 10 * base["tokens_per_param"]


def test_chapter10_repeats_and_precision():
    """§10.7's two corrections, both quoted in the prose."""
    import math
    from arith.scaling_budget import repeat_value, effective_params
    assert round(-15.4 * math.log(0.5), 1) == 10.7
    assert round(-15.4 * math.log(0.1), 1) == 35.5
    # 10.7 and 35.5 are themselves rounded, so the marginal value they imply
    # is 0.500 and 0.100 only to the precision the chapter prints them at
    assert abs(repeat_value(10.7)["marginal"] - 0.50) < 1e-3
    assert abs(repeat_value(35.5)["marginal"] - 0.10) < 1e-3
    N = non_embedding(MODEL_D)
    assert round(100 * (1 - effective_params(N, 16) / N), 4) == 0.0000
    assert round(100 * (1 - effective_params(N, 8) / N), 2) == 0.07
    assert round(100 * (1 - effective_params(N, 4) / N), 1) == 2.6


def test_chapter10_run_table_is_reproducible():
    """§10.1's table is synthetic and the author note says so.  What the note
    also claims is that it regenerates exactly, and an unreproducible synthetic
    table would be worse than no table."""
    import os
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    from figs.data.make_scaling_runs import table, render, CSV
    with open(CSV) as fh:
        assert render(table()) == fh.read()


def test_chapter10_fit_fragility():
    """§10.1 and §10.5.  Slow, so the restart ensemble is not re-run here: what
    is pinned is the sampling experiment's qualitative claim, which is the one
    the prose leans on.  Both exponents are recovered an order of magnitude
    better than either coefficient under the same noise."""
    from arith.scaling_budget import sampling_spread
    # forty draws, not the two hundred the prose reports, so the spans come
    # out smaller than the printed 3.0 and 3.8: a range is a maximum over
    # draws and grows with the number of them.  The inequality between the
    # exponents and the coefficients is what is being pinned, not the width.
    s = sampling_spread(n_draws=40)
    assert s["alpha"]["span"] < 1.3 and s["beta"]["span"] < 1.3
    assert s["A"]["span"] > 1.8 and s["B"]["span"] > 1.8
    assert s["A"]["span"] > 4 * (s["alpha"]["span"] - 1)
    assert s["B"]["span"] > 4 * (s["beta"]["span"] - 1)
    assert abs(s["alpha"]["mean"] - REFIT_2024["alpha"]) < 0.01
    assert abs(s["beta"]["mean"] - REFIT_2024["beta"]) < 0.01


def test_chapter11_kv_cache_table():
    """A-11.1, every cell.  The five rows are two formulas: a per-head cache of
    2 L n_kv d_h p_b, and a latent cache of (d_c + d_r) L p_b with no factor of
    two, because K and V are both reconstructed from the same latent."""
    from arith.kv_cache import schemes, cache_bytes, KiB, MiB
    from arith.accelerators import GiB
    r = schemes()
    assert r["D GQA"]["bytes"] == 131_072 == 128 * KiB
    assert r["D MHA"]["bytes"] == 524_288 == 512 * KiB
    assert r["D MQA"]["bytes"] == 16_384 == 16 * KiB
    assert r["S MLA"]["bytes"] == 70_272
    assert round(r["S MLA"]["bytes"] / KiB, 4) == 68.625
    assert r["S MHA"]["bytes"] == 3_997_696
    assert r["S MHA"]["bytes"] / MiB == 3.8125
    # the three punchline ratios
    assert round(r["S MHA"]["bytes"] / r["S MLA"]["bytes"], 2) == 56.89
    assert round(r["D MHA"]["bytes"] / r["S MLA"]["bytes"], 2) == 7.46
    assert round(r["D GQA"]["bytes"] / r["S MLA"]["bytes"], 2) == 1.87
    # every entry of the table, at the three printed context lengths
    expect = {
        "D MHA": (4.0, 16.0, 64.0), "D GQA": (1.0, 4.0, 16.0),
        "D MQA": (0.125, 0.5, 2.0), "S MHA": (30.5, 122.0, 488.0),
        "S MLA": (0.5361, 2.145, 8.578),
    }
    for name, want in expect.items():
        for s, w in zip((8192, 32768, 131072), want):
            got = cache_bytes(r[name]["bytes"], s) / GiB
            assert abs(got / w - 1) < 5e-4, (name, s, got, w)
    # the S-MHA row must be in the ratio 1 : 4 : 16 like every other row, which
    # is what the blueprint's 29.8 and 119 violated
    g = [cache_bytes(r["S MHA"]["bytes"], s) for s in (8192, 32768, 131072)]
    assert g[1] == 4 * g[0] and g[2] == 16 * g[0]


def test_chapter11_decode_intensity_and_roofline():
    """D-11.1 steps 7 to 9, and M-11.1's quantitative claim."""
    from arith.kv_cache import (decode_intensity, decode_flops, cache_bytes,
                                schemes, latency_floor, BF16, FP8)
    from arith.accelerators import DEFAULT, GiB
    d = MODEL_D
    assert decode_intensity(d.h, d.n_kv, BF16) == 4.0
    assert decode_intensity(d.h, d.h, BF16) == 1.0
    assert decode_intensity(d.h, 1, BF16) == 32.0
    assert decode_intensity(d.h, d.n_kv, FP8) == 8.0
    # independent of s, L and d_h: check by dividing the two counts directly
    for s in (1024, 8192, 131072):
        for n_kv in (1, 8, 32):
            flops = decode_flops(d.L, d.h, d.d_h, s)
            byts = 2 * d.L * n_kv * d.d_h * s * BF16
            assert abs(flops / byts - decode_intensity(d.h, n_kv)) < 1e-12
    assert round(DEFAULT.balance, 1) == 295.2
    assert round(DEFAULT.balance / 4.0, 1) == 73.8
    b = cache_bytes(schemes()["D GQA"]["bytes"], 131072)
    assert b / GiB == 16.0
    assert round(1e3 * latency_floor(b), 2) == 5.13
    assert round(1 / latency_floor(b)) == 195


def test_chapter11_capacity_plan():
    """The two design-consequence numbers, which come from one division and
    must be quoted the same way."""
    from arith.kv_cache import concurrency, schemes
    from arith.accelerators import DEFAULT
    w = 2 * total_params(MODEL_D)
    assert round(w / (1 << 30), 3) == 14.958
    assert round(DEFAULT.capacity_gib, 1) == 131.3
    gqa = schemes()["D GQA"]["bytes"]
    long = concurrency(gqa, 131_072, w)
    short = concurrency(gqa, 8_192, w)
    assert round(long["free_gib"], 2) == round(short["free_gib"], 2) == 116.36
    assert round(long["exact"], 2) == 7.27 and long["admit"] == 7
    assert round(short["exact"], 2) == 116.36 and short["admit"] == 116
    # E-11.6: fp8 at s = 32768, b = 16 fits on an 80 GB part alongside weights
    fp8 = schemes(p_b=1)["D GQA"]["bytes"]
    assert fp8 == 65_536
    assert (fp8 * 32_768 * 16) / (1 << 30) == 32.0
    assert fp8 * 32_768 * 16 + w < 80e9


def test_chapter11_rope_obstruction():
    """D-11.3 step 5's counterexample, and E-11.4(b)'s repair.

    The 896 GiB is the whole argument for the decoupled split, so it is pinned;
    and so is the fact that a delta-free absorbed form does exist, in exactly
    d_h terms, because the chapter corrects the blueprint on that point."""
    import numpy as np
    from arith.model_s import MODEL_S as s
    elements = s.d * s.d_c * s.extended_context
    assert round(elements / 1e11, 2) == 4.81
    assert round(2 * elements / (1 << 30)) == 896
    assert s.d_c + s.d_r == 576
    assert (s.d_c + s.d_r) * s.L * 2 == 70_272

    d_h = 64                                   # small stand-in, same algebra
    def R(delta):
        th = 10000.0 ** (-np.arange(0, d_h, 2) / d_h) * delta
        M = np.zeros((d_h, d_h))
        for p, (c, sn) in enumerate(zip(np.cos(th), np.sin(th))):
            M[2*p:2*p+2, 2*p:2*p+2] = [[c, sn], [-sn, c]]
        return M
    rng = np.random.default_rng(11)
    W_q = rng.normal(size=(128, d_h)); W_uk = rng.normal(size=(32, d_h))
    assert not np.allclose(W_q @ R(0) @ W_uk.T, W_q @ R(7) @ W_uk.T)
    # exactly d_h fixed matrices, d_h/2 cosines and d_h/2 sines
    th = 10000.0 ** (-np.arange(0, d_h, 2) / d_h)
    C = [np.zeros((d_h, d_h)) for _ in range(d_h // 2)]
    S = [np.zeros((d_h, d_h)) for _ in range(d_h // 2)]
    for p in range(d_h // 2):
        C[p][2*p, 2*p] = C[p][2*p+1, 2*p+1] = 1.0
        S[p][2*p, 2*p+1] = 1.0; S[p][2*p+1, 2*p] = -1.0
    assert len(C) + len(S) == d_h
    for delta in (0, 7, 129):
        rebuilt = sum(np.cos(th[p]*delta)*C[p] + np.sin(th[p]*delta)*S[p]
                      for p in range(d_h // 2))
        assert np.allclose(rebuilt, R(delta), atol=1e-12), delta


def test_chapter11_absorption_identity():
    """D-11.2, both halves.  Associativity, checked rather than asserted."""
    import numpy as np
    rng = np.random.default_rng(12)
    d, d_c, d_h = 256, 64, 32
    X = rng.normal(size=(7, d)); C = rng.normal(size=(11, d_c))
    W_q = rng.normal(size=(d, d_h)); W_uk = rng.normal(size=(d_c, d_h))
    assert np.allclose((X @ W_q) @ (C @ W_uk).T, X @ (W_q @ W_uk.T) @ C.T,
                       atol=1e-10)
    assert (W_q @ W_uk.T).shape == (d, d_c)
    # step 8: the value side folds into W_O and the mix happens on the latents
    W_uv = rng.normal(size=(d_c, d_h)); W_o = rng.normal(size=(d_h, d))
    A = rng.dirichlet(np.ones(11), size=7)
    assert np.allclose((A @ (C @ W_uv)) @ W_o, (A @ C) @ (W_uv @ W_o), atol=1e-10)
    assert (W_uv @ W_o).shape == (d_c, d)


def test_chapter11_online_softmax_is_exact():
    """D-11.4, and its failure mode.  The recurrence agrees with the unblocked
    reference to rounding for every partition, including ragged ones and ones
    visited out of order, and is never bitwise equal to it."""
    import numpy as np
    rng = np.random.default_rng(7)
    s, d_h = 4096, 128
    z = rng.normal(scale=3.0, size=s); v = rng.normal(size=(s, d_h))

    def online(order, n_blocks):
        m, l, o = -1e30, 0.0, np.zeros(d_h)
        for B in np.array_split(order, n_blocks):
            if len(B) == 0:
                continue
            mn = max(m, z[B].max()); r = np.exp(m - mn)
            w = np.exp(z[B] - mn)
            l = r*l + w.sum(); o = r*o + (w[:, None]*v[B]).sum(0); m = mn
        return o / l

    w = np.exp(z - z.max())
    ref = (w[:, None]*v).sum(0) / w.sum()
    idx = np.arange(s)
    for n in (1, 2, 3, 7, 64, 512):
        assert np.abs(online(idx, n) - ref).max() < 1e-12, n
    assert np.abs(online(rng.permutation(s), 64) - ref).max() < 1e-12
    assert not np.array_equal(online(idx, 64), ref)


def test_chapter11_section_118_costs():
    """The two crossovers of section 11.8, and the indexer width at which no
    context length works."""
    from arith.kv_cache import linear_attention, dsa_breakeven, window_horizon
    d = MODEL_D
    for m, xover in ((d.d_h, 256), (d.d, 8192)):
        r = linear_attention(m)
        assert r["flop_crossover"] == m
        assert r["state_equals_cache_at"] == xover == 2 * m
        assert r["state_elements_per_layer"] == m * d.d
    assert linear_attention(d.d)["state_bytes"] == 1 << 30
    # top-k selection: cheap has a precise meaning, and it is h_I d_I << 2d
    assert round(dsa_breakeven(2048, 4, 128)["breakeven_s"]) == 2185
    assert dsa_breakeven(2048, 64, 128)["indexer_width"] == 2 * d.d
    assert dsa_breakeven(2048, 64, 128)["breakeven_s"] == float("inf")
    # the saving at 128k, quoted in the prose as 92% of full attention
    s, k = 131_072, 2048
    full, sel, idx = 4*s*s*d.d, 4*s*k*d.d, 2*s*s*4*128
    assert round(100 * (1 - (sel + idx)/full)) == 92
    assert window_horizon(4096, d.L) == d.extended_context == 131_072


def test_chapter11_io_ratio():
    """Section 11.7: a modest constant, not a change of exponent."""
    from arith.accelerators import io_advantage, sram_elements, H200
    assert sram_elements(H200) == 116_736
    assert round(io_advantage(MODEL_D.d_h, H200), 2) == 7.12
    # the exponent on s is unchanged, which is the point the section makes
    assert io_advantage(MODEL_D.d_h) < 10


def test_chapter11_finite_rank_gap():
    """Section 11.8's closing claim, measured: the softmax kernel matrix is
    full rank, and the best rank-m approximation gets worse as rows sharpen."""
    import numpy as np
    rng = np.random.default_rng(21)
    d_h, s = 128, 512
    K = rng.normal(size=(s, d_h))
    errs, entropies = [], []
    for scale in (0.5, 1.0, 2.0):
        Q = rng.normal(size=(s, d_h)) * scale
        M = np.exp(Q @ K.T / np.sqrt(d_h))
        P = M / M.sum(1, keepdims=True)
        sv = np.linalg.svd(M, compute_uv=False)
        assert (sv > sv[0] * 1e-8).sum() == s, "the kernel matrix is not full rank"
        U, sv, Vt = np.linalg.svd(M, full_matrices=False)
        Mm = (U[:, :64] * sv[:64]) @ Vt[:64]
        Pm = Mm / np.maximum(Mm.sum(1, keepdims=True), 1e-300)
        errs.append(np.abs(Pm - P).mean())
        entropies.append(-(P * np.log(P + 1e-300)).sum(1).mean())
    assert entropies[0] > entropies[1] > entropies[2], entropies
    assert errs[0] < errs[1] < errs[2], errs


def test_chapter11_topk_recall_measurement():
    """E-11.14's bracket, on the committed scores.  A measurement, not a bound
    on output quality, and the test says so by pinning only the ordering."""
    import os
    import numpy as np
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "figs", "data", "ch11_scores.npz")
    if not os.path.exists(path):
        return
    z = np.load(path)
    raw, Q, K = (z["scores"].astype(np.float64), z["Q"].astype(np.float64),
                 z["K"].astype(np.float64))
    h, s, d_h = Q.shape
    mask = np.triu(np.full((s, s), -np.inf), 1)
    true = raw + mask

    def recall(approx, k):
        hits = rows = 0
        for i in range(h):
            for r in range(s):
                valid = np.isfinite(true[i, r])
                if valid.sum() < k:
                    continue
                ix = np.flatnonzero(valid)
                t = set(ix[np.argsort(-true[i, r][ix])[:k]].tolist())
                a = set(ix[np.argsort(-approx[i, r][ix])[:k]].tolist())
                hits += len(t & a); rows += 1
        return hits / (rows * k)

    best = np.empty_like(raw)
    for i in range(h):
        U, sv, Vt = np.linalg.svd(raw[i], full_matrices=False)
        best[i] = (U[:, :4] * sv[:4]) @ Vt[:4]
    assert round(recall(best + mask, 8), 3) == 0.914
    rng = np.random.default_rng(5)
    blind = np.empty_like(raw)
    for i in range(h):
        P = rng.normal(size=(d_h, 4)) / np.sqrt(d_h)
        blind[i] = (Q[i] @ P) @ (K[i] @ P).T / 2.0
    assert recall(blind + mask, 8) < 0.6


def test_chapter12_expert_ledger():
    """A-12.1's parameter accounting, expert by expert.  Model S totals
    670.408 B and not a nominal 671 B, for the reason recorded in
    arith/model_s.py: its attention block is a documented MLA parameterisation
    rather than a transcription of a checkpoint."""
    from arith.model_s import (MODEL_S, totals, expert_params,
                               moe_layer_params, moe_layer_active)
    c = MODEL_S
    n_moe = c.L - c.dense_layers
    assert n_moe == 58
    assert expert_params(c) == 3 * c.d * c.d_expert == 44_040_192
    assert round(expert_params(c) / 1e6, 2) == 44.04
    assert moe_layer_params(c) == (c.E + c.shared) * expert_params(c)
    assert round(moe_layer_params(c) / 1e9, 2) == 11.32
    assert round(moe_layer_params(c) * n_moe / 1e9, 1) == 656.5
    t = totals(c)
    assert round(t["total"] / 1e9, 3) == 670.408
    assert round(moe_layer_active(c) * n_moe / 1e9, 2) == 22.99
    assert round((t["active"] - moe_layer_active(c) * n_moe) / 1e9, 2) == 13.95
    assert round(t["active"] / 1e9, 2) == 36.93
    assert round(100 * t["active"] / t["total"], 1) == 5.5


def test_chapter12_two_numbers_two_resources():
    """M-12.1, quantitatively.  Compute follows active parameters and memory
    follows total, and the whole misconception is treating one number as
    standing for both."""
    from arith.model_s import MODEL_S, totals, flops_per_token, weight_bytes, devices
    t = totals(MODEL_S)
    fl, mem = flops_per_token(t["active"]), weight_bytes(t["total"])
    assert round(fl / 1e9, 1) == 73.9
    assert round(flops_per_token(t["total"]) / 1e9) == 1341
    assert round(mem / 1e9) == 1341
    # the same quantity in two units, which is the mnemonic the box prints
    assert flops_per_token(t["total"]) == weight_bytes(t["total"])
    assert round(t["total"] / t["active"], 1) == 18.2
    assert devices(t["total"], 80e9) == 17 and devices(t["total"], 141e9) == 10
    # against Model D: 5.3x the compute, 83.5x the memory
    d_active = non_embedding(MODEL_D)
    assert round(fl / flops_per_token(d_active), 1) == 5.3
    assert round(mem / weight_bytes(total_params(MODEL_D)), 1) == 83.5


def test_chapter12_capacity_carries_the_k():
    """Section 12.3.  Dropping the k under-sizes every buffer by a factor of
    k, which for Model S is eight, and the design consequence turns on it."""
    from arith.model_s import MODEL_S, expert_capacity, dropped_fraction
    c = MODEL_S
    r = expert_capacity(8192, c.E, c.k, 1.25)
    assert r["mean_load"] == c.k * 8192 / c.E == 256
    assert r["capacity"] == 320
    assert abs(r["slack_fraction"] - 0.2) < 1e-12
    # without the k it would be 32, eight times too small
    assert 8192 / c.E == 32 and r["mean_load"] / 32 == c.k
    # at c = 1 any imbalance drops tokens; a perfectly balanced load drops none
    import numpy as np
    rng = np.random.default_rng(3)
    loads = np.full(c.E, 256.0)
    assert dropped_fraction(loads, 256.0) == 0.0
    skew = loads * (1 + 0.55 * rng.standard_normal(c.E))
    skew = np.clip(skew, 0, None) * loads.sum() / max(skew.sum(), 1e-9)
    assert dropped_fraction(skew, 256.0) > dropped_fraction(skew, 320.0)


def test_chapter12_routing_gradient_is_zero_off_the_selection():
    """D-12.1, and the step-6 correction.  The blueprint closes the argument
    with an identity that is true but does not apply; what closes it is that
    the renormalised gate is homogeneous of degree zero in the selected gates,
    so its radial derivative vanishes by Euler's theorem."""
    import numpy as np
    rng = np.random.default_rng(12)
    E, k, d = 16, 4, 32
    x = rng.normal(size=d)
    Wr = rng.normal(size=(d, E)) / np.sqrt(d)
    Ex = rng.normal(size=(E, d))
    z = x @ Wr
    g = np.exp(z - z.max()); g /= g.sum()
    T = np.argsort(-z)[:k]
    S = g[T].sum()

    # the contraction that closes the argument is exactly zero, for every i
    for i in range(k):
        contracted = sum(((1.0 if i == j else 0.0) * S - g[T][i]) / S ** 2 * g[T][j]
                         for j in range(k))
        assert abs(contracted) < 1e-15
    # the blueprint's stated quantity is not
    blueprint = sum(g[T][i] * (S - g[T][i]) / S ** 2 for i in range(k))
    assert abs(blueprint) > 0.1, blueprint

    def out(zz, renorm):
        gg = np.exp(zz - zz.max()); gg /= gg.sum()
        gh = gg[T] / gg[T].sum() if renorm else gg[T]
        return (gh[:, None] * Ex[T]).sum(0)

    h = 1e-7
    unsel = [m for m in range(E) if m not in set(T.tolist())]

    def slope(m, renorm):
        up, dn = z.copy(), z.copy()
        up[m] += h; dn[m] -= h
        return np.abs(out(up, renorm) - out(dn, renorm)).max() / (2 * h)

    assert max(slope(m, True) for m in unsel) < 1e-6
    assert max(slope(m, True) for m in T) > 0.1
    assert max(slope(m, False) for m in unsel) > 1e-3


def test_chapter12_aux_loss_identity_and_gradient():
    """D-12.2.  The identity holds only at P = f, and the gradient's sign rule
    is the mechanism the whole section is about."""
    import numpy as np
    rng = np.random.default_rng(21)
    E, T, k, a = 64, 4096, 6, 0.001
    z = rng.normal(size=(T, E)) * 1.5
    g = np.exp(z - z.max(1, keepdims=True)); g /= g.sum(1, keepdims=True)
    sel = np.argsort(-z, axis=1)[:, :k]
    f = np.bincount(sel.ravel(), minlength=E) / (T * k)
    assert abs(f.sum() - 1) < 1e-12 and abs(g.mean(0).sum() - 1) < 1e-12
    assert abs(a * E * float(f @ f) - (a + a * E ** 2 * float(np.var(f)))) < 1e-15
    # the minimum is alpha_aux and does not depend on E, which is why it transfers
    for e in (16, 64, 256):
        u = np.full(e, 1.0 / e)
        assert abs(a * e * float(u @ u) - a) < 1e-15
    # the gradient, against finite differences with f held constant
    ana = (a * E / T) * g * (f[None, :] - (g * f[None, :]).sum(1, keepdims=True))

    def loss(zz):
        gg = np.exp(zz - zz.max(1, keepdims=True)); gg /= gg.sum(1, keepdims=True)
        return a * E * float(f @ gg.mean(0))

    h = 1e-6
    for t, j in ((0, 0), (5, 3), (100, 40)):
        up, dn = z.copy(), z.copy()
        up[t, j] += h; dn[t, j] -= h
        assert abs((loss(up) - loss(dn)) / (2 * h) - ana[t, j]) < 1e-12


def test_chapter12_controller_band_and_ripple():
    """D-12.3.  The linearised band, and the limit-cycle amplitude, which is
    u * g_p and not the blueprint's dimensionally impossible u / g_p."""
    import numpy as np
    E, k, d, N = 32, 4, 64, 4096
    rng = np.random.default_rng(31)
    Z = rng.normal(size=(N, d)) @ (rng.normal(size=(d, E)) / np.sqrt(d))

    def loads(gamma):
        sel = np.argpartition(-(Z + gamma), k, axis=1)[:, :k]
        return np.bincount(sel.ravel(), minlength=E) / (N * k)

    h = 0.02
    g_p = float(np.mean([(loads(np.eye(E)[i] * h)[i] - loads(-np.eye(E)[i] * h)[i])
                         / (2 * h) for i in range(E)]))
    assert 0.03 < g_p < 0.08, g_p

    def run(u, steps, sign=True):
        gamma = np.zeros(E); hist = []
        for _ in range(steps):
            e = 1.0 / E - loads(gamma)
            hist.append(np.abs(e).max())
            gamma = gamma + u * (np.sign(e) if sign else e)
        return np.array(hist)

    # the sign rule limit-cycles at an amplitude of order u * g_p
    for u in (0.03, 0.1, 0.3):
        tail = run(u, 900)[-200:].mean()
        assert 0.6 < tail / (u * g_p) < 1.7, (u, tail, u * g_p)
    # the proportional form degrades through the band and does not diverge,
    # because top-k selection saturates
    inside = run(1.0 / g_p * 0.5, 300, sign=False)[-50:].mean()
    edge = run(1.0 / g_p * 1.85, 300, sign=False)[-50:].mean()
    beyond = run(1.0 / g_p * 2.3, 300, sign=False)[-50:].mean()
    assert inside < edge < beyond
    assert beyond < 0.5, "a saturating plant cannot diverge"


def test_chapter12_fine_graining_combinatorics():
    """Section 12.5.  Splitting each expert m ways and raising k to km leaves
    parameters and per-token work unchanged and multiplies the number of
    distinct routings by an enormous factor."""
    from math import comb
    assert comb(64, 2) == 2016
    assert comb(256, 8) == 409_663_695_276_000
    assert round(comb(256, 8) / 1e14, 1) == 4.1
    # parameters and work are linear in the width, so both are untouched
    d, d_e, E, k, m = 7168, 2048, 64, 2, 4
    assert 3 * d * d_e * E == 3 * d * (d_e // m) * (E * m)
    assert 3 * d * d_e * k == 3 * d * (d_e // m) * (k * m)


def test_chapter13_bits_per_weight():
    """M-13.1.  Nobody ships 4.000 bits per weight, and the 9% spread across
    the shipped 4-bit formats is 0.4 GB on Model D."""
    from arith.quant_formats import FORMATS, double_quantisation_saving
    by = {f.name: f.bits for f in FORMATS}
    assert by["int4, fp16 scale per 128"] == 4.125
    assert by["int4, fp16 scale and zero per 128"] == 4.25
    assert by["MXFP4 (E8M0 scale per 32)"] == 4.25
    assert by["NVFP4 (FP8 scale per 16)"] == 4.5
    assert round(by["NF4, double quantisation"], 4) == 4.127
    named = [f.bits for f in FORMATS if f.named]
    assert round(100 * (max(named) / min(named) - 1), 1) == 9.1
    n = total_params(MODEL_D)
    assert round(n * (max(named) - min(named)) / 8 / 1e9, 3) == 0.376
    # E-13.9
    assert round(double_quantisation_saving(), 5) == 0.37305
    assert round(n * double_quantisation_saving() / 8 / 1e6) == 374
    # and the quantised cache of section 13.2
    d = MODEL_D
    vals = 2 * d.L * d.n_kv * d.d_h
    assert vals == 65_536
    for bits, kb in ((16, 131.072), (8.125, 66.56), (4.125, 33.792)):
        assert abs(vals * bits / 8 / 1e3 - kb) < 0.02, bits
    assert round((16 / 8.125), 2) == 1.97 and round((16 / 4.125), 2) == 3.88


def test_chapter13_snr_and_the_optimal_clamp():
    """D-13.1 step 8, and the correction the uniform-error model cannot see:
    the best clamp depends on the bit width, because clipping is the other
    error source and the formula ignores it."""
    import numpy as np
    from arith.quant_formats import snr_db
    assert round(snr_db(4, 8.0), 2) == 16.81
    assert round(snr_db(8, 8.0), 2) == 40.89
    assert round(snr_db(4, 6.0), 2) == 19.31
    rng = np.random.default_rng(13)
    x = rng.standard_normal(400_000)

    def measured(b_q, clip):
        s = 2 * clip / (2 ** b_q - 1)
        z = np.round(clip / s)
        xh = s * (np.clip(np.round(x / s) + z, 0, 2 ** b_q - 1) - z)
        return 10 * np.log10(float((x ** 2).mean() / ((x - xh) ** 2).mean()))

    grid = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
    best4 = max(grid, key=lambda c: measured(4, c))
    best8 = max(grid, key=lambda c: measured(8, c))
    assert best4 == 2.5 and best8 == 4.0, (best4, best8)
    assert measured(4, 2.5) > measured(4, 4.0) + 2.0


def test_chapter13_rotation_and_incoherence():
    """D-13.2.  The bound, the bits, and the reason a fixed Hadamard is not a
    random rotation."""
    import numpy as np
    d = 4096
    assert round(np.sqrt(d / (2 * np.log(d))), 2) == 15.69
    assert round(float(np.log2(np.sqrt(d / (2 * np.log(d))))), 2) == 3.97
    assert round(float(np.sqrt(2 * np.log(d))), 2) == 4.08

    H = np.array([[1.0]])
    while H.shape[0] < d:
        H = np.block([[H, H], [H, -H]])
    rng = np.random.default_rng(23)
    inc = lambda v: np.sqrt(len(v)) * np.abs(v).max() / np.linalg.norm(v)

    # a realistically spiked row: the gain is 2.8 bits, not the 3.97 bound
    x = rng.standard_normal(d)
    x[rng.choice(d, 12, replace=False)] *= 20.0
    Q = (H * rng.choice([-1.0, 1.0], d)[:, None]) / np.sqrt(d)
    before, after = inc(x), inc(x @ Q)
    assert before > 15 and after < 4.5
    assert 2.0 < np.log2(before / after) < 3.5

    # exactness, and RMSNorm equivariance
    W = rng.standard_normal((d, 64)) / np.sqrt(d)
    assert np.abs(x @ W - (x @ Q) @ (Q.T @ W)).max() < 1e-9
    rms = lambda u: u / np.sqrt((u ** 2).mean())
    assert np.abs(rms(x @ Q) - rms(x) @ Q).max() < 1e-9

    # a vector aligned with a Hadamard row defeats a fixed Hadamard entirely
    row = H[7] / np.sqrt(d)
    assert inc(row @ (H / np.sqrt(d))) > 0.9 * np.sqrt(d)
    worst = max(inc(row @ ((H * np.random.default_rng(s).choice([-1.0, 1.0], d)[:, None])
                           / np.sqrt(d))) for s in range(8))
    assert worst < 10.0, worst


def test_chapter13_obs_closed_form():
    """D-13.3.  The Lagrange solution against a brute-force constrained solve,
    and the cost identity."""
    import numpy as np
    rng = np.random.default_rng(31)
    n, m = 64, 256
    X = rng.standard_normal((m, n))
    H = 2 * X.T @ X
    H = H + 0.01 * np.mean(np.diag(H)) * np.eye(n)
    Hinv = np.linalg.inv(H)
    for j in (0, 7, 63):
        dj = 0.37
        D_closed = -(dj / Hinv[j, j]) * Hinv[:, j]
        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = H; A[:n, n] = np.eye(n)[j]; A[n, :n] = np.eye(n)[j]
        b = np.zeros(n + 1); b[n] = -dj
        assert np.abs(D_closed - np.linalg.solve(A, b)[:n]).max() < 1e-12
        assert abs(0.5 * D_closed @ H @ D_closed - dj ** 2 / (2 * Hinv[j, j])) < 1e-12
        assert abs(D_closed[j] + dj) < 1e-12, "the constraint must be met exactly"


def test_chapter13_lora_scaling_is_rank_free_at_one_over_sqrt_r():
    """D-13.4 steps 5 to 7, and the correction to the blueprint.  Delta y goes
    as gamma^2 r, so the rank-invariant scaling is gamma proportional to
    1/sqrt(r), not the conventional alpha/r."""
    import numpy as np
    d = k = 4096
    ranks = (4, 16, 64, 256)

    def step1(r, gamma, seed, adam=False):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(d)
        g = rng.standard_normal(k) / np.sqrt(k)
        B = rng.standard_normal((d, r)) / np.sqrt(d)
        u = x @ B
        gA = gamma * np.outer(u, g)
        dA = -1e-3 * (np.sign(gA) if adam else gA)
        return float(np.linalg.norm(gamma * (u @ dA)))

    def sweep(gfun, adam=False):
        return [np.mean([step1(r, gfun(r), s, adam) for s in range(24)])
                for r in ranks]

    # ||u||^2 is linear in r, which is what makes the exponent what it is.
    # One draw of B has relative spread sqrt(2/r), so this is averaged: the
    # claim is about the expectation, not about any particular adapter.
    x = np.random.default_rng(1).standard_normal(d)
    for r in ranks:
        got = np.mean([float((x @ B) @ (x @ B)) for B in
                       (np.random.default_rng(100 + s).standard_normal((d, r))
                        / np.sqrt(d) for s in range(16))])
        assert abs(got / (r * float(x @ x) / d) - 1) < 0.12, (r, got)

    flat = sweep(lambda r: 8.0 / np.sqrt(r))
    assert 0.8 < flat[-1] / flat[0] < 1.25, flat
    grows = sweep(lambda r: 2.0)
    assert grows[-1] / grows[0] > 30, grows          # gamma constant: goes as r
    falls = sweep(lambda r: 16.0 / r)
    assert falls[-1] / falls[0] < 0.05, falls        # alpha/r: goes as 1/r


def test_chapter13_finetune_memory_table():
    """A-13.1, every cell, including the grouped-query adapter trap."""
    from arith.model_d import (lora_params, activation_bytes, finetune_memory,
                               smallest_device)
    lp = lora_params(MODEL_D, 16)
    assert lp["per_layer"] == 425_984
    assert lp["total"] == 13_631_488
    assert round(100 * lp["total"] / total_params(MODEL_D), 3) == 0.170
    assert lp["mha_total"] == 16_777_216
    assert lp["over_count"] == 3_145_728
    assert round(100 * (lp["mha_total"] / lp["total"] - 1)) == 23
    assert round(16 * lp["total"] / 1e6) == 218
    assert round(activation_bytes(MODEL_D, 8, 4096) / 1e9, 2) == 8.59

    rows = finetune_memory(b=8, s=4096)
    assert round(rows["full"]["state"] / 1e9, 1) == 128.5
    assert round(rows["full"]["total"] / 1e9, 1) == 137.1
    assert round(rows["lora"]["weights"] / 1e9, 2) == 16.06
    assert round(rows["lora"]["total"] / 1e9, 1) == 24.9
    assert round(rows["qlora"]["weights"] / 1e9, 2) == 4.14
    assert round(rows["qlora"]["total"] / 1e9, 1) == 13.0
    assert smallest_device(rows["lora"]["total"]) == "1 x 40 GB"
    assert smallest_device(rows["qlora"]["total"]) == "1 x 24 GB"
    assert smallest_device(rows["full"]["total"]) == "1 x 141 GB"

    # E-13.8: at long context it is LoRA and QLoRA that converge, not LoRA and
    # full fine-tuning, because quantising the base stops mattering once
    # activations dominate
    long = finetune_memory(b=8, s=32768)
    assert round(long["full"]["total"] / long["lora"]["total"], 1) == 2.3
    assert round(long["lora"]["total"] / long["qlora"]["total"], 2) == 1.16
    assert round(rows["lora"]["total"] / rows["qlora"]["total"], 2) == 1.92


def test_chapter13_nf4_levels_and_ordering():
    """The honest ordering: int4 worst, NF4 in between, Lloyd-Max best, and
    NF4 is what ships because its table is fixed."""
    import numpy as np
    from arith.quant_formats import nf4_levels
    lv = nf4_levels()
    assert lv.shape == (16,) and np.any(lv == 0.0)
    assert abs(lv.min() + 1) < 1e-12 and abs(lv.max() - 1) < 1e-12
    gaps = np.diff(lv)
    assert round(gaps.min(), 4) == 0.0796 and round(gaps.max(), 4) == 0.3038

    rng = np.random.default_rng(53)
    x = rng.standard_normal(100_000)
    xn = x / np.abs(x).max()
    to = lambda L, v: L[np.abs(v[:, None] - L[None, :]).argmin(1)]
    mse_int4 = float(((xn - to(np.linspace(-1, 1, 16), xn)) ** 2).mean())
    mse_nf4 = float(((xn - to(lv, xn)) ** 2).mean())
    assert mse_nf4 < 0.6 * mse_int4, (mse_nf4, mse_int4)
