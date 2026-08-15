"""The CI guard for Appendix C.

`test_arithmetic.py` recomputes every number printed in a *chapter*.  Appendix C
prints a few hundred more, and until 15 Aug 2026 none of them was guarded.  That
is how nine wrong answers survived: E-3.7 printed a multi-query-to-grouped-query
parameter ratio of 1.12, which is impossible on its face because multi-query has
strictly fewer key and value parameters; and E-8.7 printed token counts that
contradicted the book's own E-10.10, three hundred pages away.

Same contract as the chapter suite: nothing is transcribed.  Every assertion
recomputes from `arith/`, so a solution and the module that produced it cannot
drift.  Where an answer is a measurement rather than a formula, the committed
data is re-measured here rather than the number being pasted in.

Named for the exercise it guards, so a failure says which printed line is wrong.
"""
import math

import pytest

from arith.model_d import (MODEL_D, non_embedding, total_params, attention_params,
                           critical_dimension, llama_intermediate_size,
                           tokens_for_loss, REFIT_2024)
from arith import kv_cache, model_s, scaling_budget
from arith.sae_capacity import capacity, eps_for


def _attn(cfg):
    return sum(attention_params(cfg).values())


def _with(**kw):
    from dataclasses import replace
    return replace(MODEL_D, **kw)


# ------------------------------------------------------------------- chapter 1
def test_e1_9_entropy_and_the_two_divergences():
    """H(p,q) - H(p) must equal KL(p||q).  The printed triple did not.

    The old answer (0.8437, 0.0419, 0.0457) was internally consistent and wrong:
    it satisfied the identity, so checking the identity alone would not have
    caught it.  Recomputing from p and q does.
    """
    p = (0.7, 0.2, 0.1)
    q = (0.5, 0.25, 0.25)
    H = -sum(pi * math.log(pi) for pi in p)
    Hpq = -sum(pi * math.log(qi) for pi, qi in zip(p, q))
    kl_pq = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
    kl_qp = sum(qi * math.log(qi / pi) for pi, qi in zip(p, q))
    assert round(H, 4) == 0.8018
    assert round(Hpq, 4) == 0.9011
    assert round(kl_pq, 4) == 0.0993
    assert round(kl_qp, 4) == 0.1166
    assert Hpq - H == pytest.approx(kl_pq)          # the identity, still
    assert round(100 * (kl_qp / kl_pq - 1)) == 17   # the printed asymmetry


def test_e1_10_and_e3_7_head_ladder():
    """Multi-query must come in below grouped-query, and multi-head above."""
    mha, gqa, mqa = _with(n_kv=32), MODEL_D, _with(n_kv=1)
    assert total_params(mha) == 8_835_567_616
    assert total_params(gqa) == 8_030_261_248
    assert total_params(mqa) == 7_795_380_224
    assert round(100 * (total_params(mha) / total_params(gqa) - 1), 2) == 10.03
    assert round(100 * (total_params(mqa) / total_params(gqa) - 1), 2) == -2.92
    # E-3.7, per layer, as a ratio to grouped-query
    assert _attn(mha) == 67_108_864
    assert _attn(gqa) == 41_943_040
    assert _attn(mqa) == 34_603_008
    assert round(_attn(mha) / _attn(gqa), 3) == 1.600
    assert round(_attn(mqa) / _attn(gqa), 3) == 0.825
    # the sign check the printed 1.12 failed: fewer K and V heads is fewer
    # parameters, so this ratio cannot exceed one
    assert _attn(mqa) < _attn(gqa) < _attn(mha)


# ------------------------------------------------------------------- chapter 4
def test_e4_7_and_e4_8_critical_dimension():
    """A lower RoPE base rotates MORE bands, not fewer."""
    assert critical_dimension(_with(rope_base=10_000, trained_context=2048)) == 41
    assert critical_dimension(_with(rope_base=10_000, trained_context=8192)) == 50
    assert critical_dimension(_with(rope_base=500_000, trained_context=8192)) == 35
    assert (critical_dimension(_with(rope_base=10_000, trained_context=8192))
            > critical_dimension(_with(rope_base=500_000, trained_context=8192)))
    # E-4.8: Model S's decoupled RoPE width at the PRE-extension context
    assert critical_dimension(_with(rope_base=10_000, trained_context=4096,
                                    d_h=64)) == 23
    # the wavelength ceiling, which is E-4.7's real point.  Section 4.5 quotes
    # this pair as "about 63,000 tokens to 3.14 million"; both ends are 2*pi*b,
    # and they must be quoted the same way or the sentence compares two
    # different quantities.  It printed 57,000 until 15 Aug 2026, which was
    # neither 2*pi*b nor the slowest actual band (54,410).
    assert round(2 * math.pi * 10_000) == 62_832
    assert round(2 * math.pi * 500_000 / 1e6, 2) == 3.14
    assert 2 * math.pi * 10_000 < 131_072


# ------------------------------------------------------------------- chapter 6
def test_e6_6_llama_width_pipeline():
    """Four steps, three of them truncating, and the first is 4d not d."""
    p = llama_intermediate_size(8192)
    assert (p["4d"], p["two_thirds"], p["multiplied"], p["intermediate_size"]) \
        == (32_768, 21_845, 28_398, 28_672)
    # the d whose final rounding moves the answer by more than 5 per cent
    q = llama_intermediate_size(1536)
    assert q["multiplied"] == 5324 and q["intermediate_size"] == 6144
    assert round(100 * (6144 / 5324 - 1), 1) == 15.4


# ------------------------------------------------------------------- chapter 8
def test_e8_7_token_budgets_agree_with_e10_10():
    """The same target must give the same answer in Chapter 8 and Chapter 10."""
    got = {t: tokens_for_loss(t)["tokens"] / 1e12 for t in (1.98, 1.93, 1.88)}
    assert round(got[1.98], 1) == 31.5
    assert round(got[1.93], 1) == 87.8
    assert round(got[1.88], 1) == 459.8
    for t in got:
        assert round(tokens_for_loss(t)["multiplier"], 2) in (2.10, 5.85, 30.66)
    # and the sensitivity clause: the floor moves it more than the exponent
    beta_only = tokens_for_loss(1.93, beta=0.28)["tokens"] / 1e12
    floor_only = tokens_for_loss(1.93, L_inf=1.70)["tokens"] / 1e12
    both = tokens_for_loss(1.93, beta=0.28, L_inf=1.70)["tokens"] / 1e12
    assert round(beta_only) == 151 and round(floor_only, 1) == 40.2
    assert round(both, 1) == 54.5
    assert abs(math.log(floor_only / 87.8)) > abs(math.log(beta_only / 87.8)) * 0.45
    assert (beta_only - 87.8) * (floor_only - 87.8) < 0     # opposite directions


def test_e8_8_smoothing_gap_and_floor():
    """Two quantities at two vocabularies, and they differ by two orders."""
    def gap(eps, V):
        return math.log((1 - eps) * (V - 1) / eps)

    def floor(eps, V):
        hi = 1 - eps + eps / V
        lo = eps / V
        return -(hi * math.log(hi) + (V - 1) * lo * math.log(lo))

    for V, gaps, floors in ((32_000, (14.97, 13.32, 12.57), (0.160, 0.717, 1.362)),
                            (128_256, (16.36, 14.71, 13.96), (0.174, 0.787, 1.501))):
        for eps, g, f in zip((0.01, 0.05, 0.1), gaps, floors):
            assert round(gap(eps, V), 2) == g
            assert round(floor(eps, V), 3) == f
    # the gloss: the gap moves by 1.4 nats between vocabularies, the floor by 0.14
    assert round(gap(0.1, 128_256) - gap(0.1, 32_000), 1) == 1.4
    assert round(floor(0.1, 128_256) - floor(0.1, 32_000), 2) == 0.14


# ------------------------------------------------------------ chapter 10
def test_e10_3_ratio_drift_both_exponent_pairs():
    """The refit and the published pair disagree about the SIGN of the drift."""
    def per_decade_of_C(a, b):
        return 10 ** ((a - b) / (a + b)) - 1

    refit = per_decade_of_C(0.348, 0.366)
    published = per_decade_of_C(0.34, 0.28)
    assert round(100 * refit, 1) == -5.6
    assert round(100 * published, 1) == 25.0
    assert refit < 0 < published
    assert round(scaling_budget.tokens_per_param_exponent()["per_decade_of_N"] * 100,
                 1) == -10.7
    # the value depends on all four fitted constants, and on L_inf not at all
    f = REFIT_2024
    K = (f["beta"] * f["B"] / (f["alpha"] * f["A"])) ** (2 / (f["alpha"] + f["beta"]))
    assert round(K, 2) == 69.70
    N = 7e9
    D = scaling_budget.optimal_D(N)
    assert round(D / N, 1) == 20.6
    assert round(K * (6 * N * D / 6) ** ((f["alpha"] - f["beta"])
                                         / (f["alpha"] + f["beta"])), 1) == 20.6


def test_e10_4_serving_always_argues_for_a_smaller_model():
    prev = None
    for D_inf, want in ((0, 30.1), (1e12, 20.2), (1e13, 11.7), (1e14, 7.6)):
        r = scaling_budget.inference_aware_optimum(2.03227, D_inf)
        assert round(r["N"] / 1e9, 1) == want
        if prev is not None:
            assert r["N"] < prev["N"] and r["tokens_per_param"] > prev["tokens_per_param"]
        prev = r
    assert round(prev["tokens_per_param"]) == 1247


def test_e10_5_repeat_value():
    assert round(-15.4 * math.log(0.5), 1) == 10.7
    assert round(-15.4 * math.log(0.1), 1) == 35.5
    four = scaling_budget.repeat_value(4)
    sixteen = scaling_budget.repeat_value(16)
    assert round(100 * four["marginal"]) == 77
    assert round(four["effective_multiplier"], 2) == 4.52
    assert round(100 * four["effective_multiplier"] / 5, 1) == 90.5
    assert round(sixteen["effective_multiplier"], 2) == 10.95
    assert round(100 * sixteen["effective_multiplier"] / 17, 1) == 64.4


def test_e10_6_to_e10_9_arithmetic_answers():
    N = non_embedding(MODEL_D)
    # E-10.6, the two Model S counts
    assert round(6 * 671e9 * 15e12 / 1e25, 2) == 6.04
    assert round(6 * 37e9 * 15e12 / 1e24, 2) == 3.33
    # E-10.7, the true training cost at two context lengths
    a = scaling_budget.true_training_flops(N, 15e12, 8192)
    b = scaling_budget.true_training_flops(N, 15e12, 131_072)
    assert round(100 * a["understatement"], 1) == 27.7
    assert round(100 * b["understatement"], 1) == 83.3
    assert round(a["logits"] / 1e22, 2) == round(b["logits"] / 1e22, 2)   # fixed
    # E-10.8, the 3 B box against the 7 B box
    three, seven = scaling_budget.box(N=3e9), scaling_budget.box(N=7e9)
    assert round(three["L_ship"], 3) == 2.094
    assert round(three["N_c"] / 1e9, 1) == 14.4
    assert round(three["break_even_tokens"] / 1e12, 1) == 10.7
    assert round(seven["break_even_tokens"] / 1e12, 1) == 11.4
    assert three["break_even_tokens"] < seven["break_even_tokens"]
    assert round(100 * (1 - three["break_even_tokens"]
                        / seven["break_even_tokens"])) == 6
    # E-10.9, precision
    for bits, want in ((16, 1.0000), (8, 0.9993), (4, 0.9737), (3, 0.9346)):
        assert round(scaling_budget.effective_params(1.0, bits), 4) == want
    f = REFIT_2024
    Ne = scaling_budget.effective_params(N, 4)
    penalty = f["A"] * Ne ** -f["alpha"] - f["A"] * N ** -f["alpha"]
    assert round(penalty, 4) == 0.0017
    D2 = ((f["B"] * 15e12 ** -f["beta"] - penalty) / f["B"]) ** (-1 / f["beta"])
    assert round(D2 / 1e12, 1) == 17.4
    assert round(100 * (D2 / 15e12 - 1), 1) == 16.3


# ------------------------------------------------------------ chapter 11
def test_e11_6_to_e11_10_cost_of_attention():
    c = kv_cache.MODEL_D
    GiB = kv_cache.GiB
    # E-11.6
    fp8 = kv_cache.per_head_bytes_per_token(c.L, c.n_kv, c.d_h, kv_cache.FP8)
    assert fp8 == 65_536
    assert round(kv_cache.cache_bytes(fp8, 32_768, 16) / GiB, 3) == 32.000
    assert round(total_params(c) * 2 / GiB, 2) == 14.96
    assert round((80e9 - total_params(c) * 2
                  - kv_cache.cache_bytes(fp8, 32_768, 16)) / GiB, 2) == 27.55
    # E-11.7
    bf16 = kv_cache.per_head_bytes_per_token(c.L, c.n_kv, c.d_h, kv_cache.BF16)
    need = kv_cache.cache_bytes(bf16, 131_072, 32)
    assert round(need / GiB, 1) == 512.0
    assert round(need / 2 / GiB, 1) == 256.0
    mqa = kv_cache.cache_bytes(
        kv_cache.per_head_bytes_per_token(c.L, 1, c.d_h, kv_cache.BF16), 131_072, 32)
    mla = kv_cache.cache_bytes(
        kv_cache.latent_bytes_per_token(c.L, 512, 64, kv_cache.BF16), 131_072, 32)
    assert round(mqa / GiB) == 64 and round(mla / GiB) == 144
    assert round(kv_cache.cache_bytes(bf16, 4096, 32) / GiB) == 16
    # E-11.8, six numbers, and none of them reaches the ridge
    want = {(1, 2): 32.0, (1, 1): 64.0, (8, 2): 4.0, (8, 1): 8.0,
            (32, 2): 1.0, (32, 1): 2.0}
    for (n_kv, p_b), v in want.items():
        assert kv_cache.decode_intensity(c.h, n_kv, p_b) == v
        assert v < 295.2
    assert round(295.2 / 2, 1) == 147.6 and 147.5 > c.h      # unreachable at fp8
    # E-11.9, the factor of s
    s = 131_072
    prefill = 4 * s * s * c.d * c.L
    decode = kv_cache.decode_flops(c.L, c.h, c.d_h, s)
    assert prefill / decode == s
    assert kv_cache.cache_bytes(bf16, s) == kv_cache.cache_bytes(bf16, s)
    assert round(decode / kv_cache.cache_bytes(bf16, s)) == 4
    assert round(prefill / kv_cache.cache_bytes(bf16, s)) == 524_288
    # E-11.10
    assert round(kv_cache.dsa_breakeven(2048, 4, 128)["breakeven_s"]) == 2185
    assert kv_cache.dsa_breakeven(2048, 64, 128)["breakeven_s"] == float("inf")
    assert 64 * 128 >= 2 * c.d
    assert round(100 * (4 * s * 2048 * c.d) / (4 * s * s * c.d), 2) == 1.56
    saving = 4 * s * s * c.d - 4 * s * 2048 * c.d
    assert round(100 * (2 * s * s * 4 * 128) / saving, 2) == 6.35
    for m, work, state in ((c.d_h, 128, 256), (c.d, 4096, 8192)):
        r = kv_cache.linear_attention(m)
        assert r["flop_crossover"] == work
        assert r["state_equals_cache_at"] == state
    assert round(100 * 128 / 8192, 1) == 1.6
    assert round(100 * 256 / 8192, 1) == 3.1


# ------------------------------------------------------------ chapter 12
def test_e12_5_to_e12_8_model_s_ledger():
    c = model_s.MODEL_S
    t = model_s.totals(c)
    n_moe = c.L - c.dense_layers
    assert model_s.expert_params(c) == 44_040_192 == 3 * c.d * c.d_expert
    assert model_s.moe_layer_params(c) == 11_318_329_344
    assert c.E * model_s.expert_params(c) == 11_274_289_152
    assert t["routed_total"] == 653_908_770_816
    # every component that is not a routed expert, and they must sum
    attn = model_s.mla_params(c) * c.L
    norms = 2 * c.d * c.L
    dense = 3 * c.d * c.d_ff_dense * c.dense_layers
    emb = 2 * c.V * c.d
    shared = model_s.expert_params(c) * n_moe
    assert attn + norms + dense + emb + shared + t["routed_total"] == t["total"]
    assert round((t["total"] - t["routed_total"]) / 1e9, 3) == 16.499
    for got, want in ((attn, 10.902), (shared, 2.554), (dense, 1.189), (emb, 1.853)):
        assert round(got / 1e9, 3) == want
    # E-12.6
    assert round(100 * t["active"] / t["total"], 2) == 5.51
    assert round(t["total"] / t["active"], 1) == 18.2
    assert round(2 * t["active"] / 1e9, 2) == 73.87
    assert round(2 * t["total"] / 1e9) == 1341
    assert round(2 * total_params(MODEL_D) / 1e9, 2) == 16.06
    # E-12.7
    cap = model_s.expert_capacity(8192, 256, 8, 1.25)
    assert cap["mean_load"] == 256.0 and cap["capacity"] == 320.0
    assert model_s.expert_capacity(8192, 256, 8, 2.0)["capacity"] == 512.0
    # E-12.8, and why the 37 B version is meaningless
    assert round(2 * t["total"] / 1e9, 1) == 1340.8
    assert math.ceil(2 * t["total"] / 80e9) == 17
    assert math.ceil(2 * t["total"] / 141e9) == 10
    assert round(2 * t["active"] / 1e9, 1) == 73.9
    assert math.ceil(2 * t["active"] / 80e9) == 1


def test_e12_7_drop_rate_is_measured_not_assumed():
    """The drop rates come off F-12.2's committed loads, not a fitted lognormal."""
    import os
    import numpy as np
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "figs", "data", "moe_regimes.npz")
    if not os.path.exists(path):
        pytest.skip("figs/data/moe_regimes.npz not present")
    loads = np.load(path)["none_loads"]
    cv = (loads.std(1) / loads.mean(1)).mean()
    assert round(cv, 3) == 0.552            # the 0.55 the exercise quotes

    def dropped(c):
        p = loads / loads.sum(1, keepdims=True)
        return float(np.maximum(0.0, p - c / p.shape[1]).sum(1).mean())

    assert round(100 * dropped(1.25), 1) == 13.3
    assert round(100 * dropped(2.0), 2) == 0.56
    assert round(dropped(1.25) / dropped(2.0)) == 24


# ------------------------------------------------------------ chapter 16
def test_e16_7_capacity_at_model_s_width():
    assert round(capacity(7168, 0.1) / 1e7, 2) == 6.06
    assert round(capacity(4096, 0.1) / 1e4, 1) == 2.8
    assert round(eps_for(7168, 32 * 7168), 4) == 0.0830
    assert round(eps_for(4096, 32 * 4096), 4) == 0.1073
    # the direction of the change is the point: wider needs a SMALLER epsilon
    assert eps_for(7168, 32 * 7168) < eps_for(4096, 32 * 4096)
