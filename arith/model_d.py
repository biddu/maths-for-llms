"""Model D — the dense 8B reference model used by every arithmetic box in
Parts I and II.  Defined by hyperparameters, never by name, so a second edition
edits this file and re-runs rather than re-checking.

    python arith/model_d.py                     full ledger
    python arith/model_d.py --embedding-share   the Chapter 2 box
    python arith/model_d.py --section attention the Chapter 3 box
    python arith/model_d.py --ffn              the Chapter 6 box
    python arith/model_d.py --backward         the Chapter 7 box
    python arith/model_d.py --loss             the Chapter 8 box
    python arith/model_d.py --optimiser        the Chapter 9 box
    python arith/model_d.py --finetune-memory  the Chapter 13 box
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    L: int = 32
    d: int = 4096
    h: int = 32
    d_h: int = 128
    n_kv: int = 8
    d_ff: int = 14336
    V: int = 128256
    rope_base: int = 500_000
    trained_context: int = 8192
    extended_context: int = 131_072
    tied: bool = False


MODEL_D = Config()


# --------------------------------------------------------------- parameters
def attention_params(c: Config) -> dict[str, int]:
    return {
        "W_Q": c.d * c.h * c.d_h,
        "W_K": c.d * c.n_kv * c.d_h,
        "W_V": c.d * c.n_kv * c.d_h,
        "W_O": c.h * c.d_h * c.d,
    }


def per_layer(c: Config) -> dict[str, int]:
    attn = sum(attention_params(c).values())
    mlp = 3 * c.d * c.d_ff          # SwiGLU: gate, up, down
    norms = 2 * c.d
    return {"attention": attn, "mlp": mlp, "norms": norms,
            "total": attn + mlp + norms}


def non_embedding(c: Config) -> int:
    return per_layer(c)["total"] * c.L + c.d      # + final norm


def embedding(c: Config) -> int:
    return c.V * c.d


def total_params(c: Config) -> int:
    n_emb = embedding(c) if c.tied else 2 * embedding(c)
    return non_embedding(c) + n_emb


# -------------------------------------------------------------------- FLOPs
def attention_flops(c: Config, s: int, causal: bool = False) -> dict[str, float]:
    proj = 2 * s * c.d * 2 * (c.h + c.n_kv) * c.d_h
    attn = 4 * s * s * c.d
    if causal:
        attn /= 2
    return {"projections": proj, "attention": attn, "total": proj + attn}


def crossover(c: Config) -> int:
    """s* where the s^2 d term equals the s d^2 term."""
    return (c.h + c.n_kv) * c.d_h


def mlp_flops(c: Config, s: int) -> float:
    return 2 * s * 3 * c.d * c.d_ff


# ------------------------------------------------------------------ reports
def _fmt(n) -> str:
    return f"{n:,}" if isinstance(n, int) else f"{n:,.2f}"


def report_embedding_share(c: Config = MODEL_D) -> None:
    e, tot = embedding(c), total_params(Config(**{**c.__dict__, "tied": False}))
    tied_tot = non_embedding(c) + e
    print(f"V*d                  {_fmt(e)}")
    print(f"2*V*d                {_fmt(2*e)}  = {200*e/tot:.2f}% of {_fmt(tot)}")
    print(f"tied total           {_fmt(tied_tot)}  ({100*e/tot:.2f}% cut)")
    print(f"bf16, one copy       {2*e/1e9:.3f} GB")


def report_attention(c: Config = MODEL_D, s: int | None = None) -> None:
    s = s or c.trained_context
    p = attention_params(c)
    for k, v in p.items():
        print(f"{k:<8} {_fmt(v)}")
    per = sum(p.values())
    n = non_embedding(c)
    print(f"per layer            {_fmt(per)}")
    print(f"x L                  {_fmt(per*c.L)}  = {100*per*c.L/n:.2f}% of {_fmt(n)}")
    f, fc = attention_flops(c, s), attention_flops(c, s, causal=True)
    print(f"\ns = {s}")
    print(f"  projections        {f['projections']/1e9:.2f} GFLOP")
    print(f"  scores QK^T        {f['attention']/2/1e9:.2f} GFLOP")
    print(f"  weighted sum AV    {f['attention']/2/1e9:.2f} GFLOP")
    print(f"  attention term     {f['attention']/1e12:.4f} TFLOP dense"
          f" / {fc['attention']/1e9:.1f} GFLOP causal")
    print(f"  per layer          {f['total']/1e12:.3f} TF dense"
          f" / {fc['total']/1e12:.3f} TF causal")
    print(f"  whole model        {fc['total']*c.L/1e12:.1f} TF causal")
    st = crossover(c)
    print(f"\ns* = (h + n_kv) d_h  {st} = {st/c.d:.2f} d ; s/s* at {s} = {s/st:.2f}")
    mha = 2 * c.h * c.d_h
    print(f"  MHA counterfactual {mha} ; causal doubles both to {2*st} and {2*mha}")


def report_full(c: Config = MODEL_D) -> None:
    pl = per_layer(c)
    for k, v in pl.items():
        print(f"{k:<12} {_fmt(v)}")
    print(f"non-embedding {_fmt(non_embedding(c))}")
    print(f"embeddings    {_fmt(2*embedding(c))}")
    print(f"total         {_fmt(total_params(c))}")
    print(f"bf16 weights  {2*total_params(c)/1e9:.2f} GB")


# ------------------------------------------------------------- RoPE bands
def rope_bands(c: Config = MODEL_D, target: int | None = None,
               alpha_y: float = 1.0, beta_y: float = 32.0) -> list[dict]:
    """One row per 2-D block: wavelength, turns completed over the trained
    context, YaRN's ramp weight, and the effective scale that weight produces.
    Chapter 4's arithmetic box and panel (c) of F-4.2 both read this."""
    import math
    target = target or c.extended_context
    s = target / c.trained_context
    rows = []
    for i in range(c.d_h // 2):
        lam = 2 * math.pi * c.rope_base ** (2 * i / c.d_h)
        turns = c.trained_context / lam
        g = min(1.0, max(0.0, (turns - alpha_y) / (beta_y - alpha_y)))
        rows.append({"i": i, "lambda": lam, "turns": turns, "gamma": g,
                     "effective_scale": 1.0 / ((1 - g) / s + g)})
    return rows


def critical_dimension(c: Config = MODEL_D) -> int:
    """Smallest i whose wavelength exceeds the trained context."""
    return next(r["i"] for r in rope_bands(c) if r["lambda"] > c.trained_context)


def report_rope_bands(c: Config = MODEL_D) -> None:
    import math
    rows = rope_bands(c)
    s = c.extended_context / c.trained_context
    print(f"{'i':>3} {'lambda':>14} {'turns':>11} {'gamma':>7} {'eff.scale':>10}")
    for r in rows:
        print(f"{r['i']:>3} {r['lambda']:>14,.2f} {r['turns']:>11,.4f}"
              f" {r['gamma']:>7.4f} {r['effective_scale']:>10.2f}")
    ext = sum(1 for r in rows if r["gamma"] >= 1 - 1e-12)
    itp = sum(1 for r in rows if r["gamma"] <= 1e-12)
    print(f"\ni* = {critical_dimension(c)} ; extrapolate {ext} pairs,"
          f" ramp {len(rows)-ext-itp} pairs, interpolate {itp} pairs")
    t = 0.1 * math.log(s) + 1
    print(f"scale s = {s:g} ; sqrt(1/t) = {t:.5f} ; 1/t = {t*t:.5f}")
    print(f"NTK b'_rope = {c.rope_base * s ** (c.d_h/(c.d_h-2)):.4e}")


# ------------------------------------------------------------ the FFN width
def llama_intermediate_size(d: int, multiplier: float = 1.3,
                            multiple_of: int = 1024) -> dict[str, int | float]:
    """Chapter 6's arithmetic box, step 3, as executable code.

    The Llama reference implementation computes `intermediate_size` in four
    steps and every one of them is integer-truncating except the last, which
    rounds up.  Written out rather than folded, because the box walks the
    reader through the same four lines and a folded version would hide where
    10922 becomes 14336.
    """
    import math
    h0 = 4 * d                                  # the classical width
    h1 = int(2 * h0 / 3)                        # the 2/3 convention, truncated
    h2 = int(multiplier * h1)                   # ffn_dim_multiplier
    h3 = multiple_of * math.ceil(h2 / multiple_of)   # rounded up, not to nearest
    return {"4d": h0, "two_thirds": h1, "multiplied": h2,
            "intermediate_size": h3, "ratio_to_two_thirds": h3 / (2 * h0 / 3)}


def ffn_budget(c: Config = MODEL_D) -> dict[str, int | float]:
    """Where Model D's non-embedding parameters actually are."""
    attn = sum(attention_params(c).values())
    mlp = 3 * c.d * c.d_ff
    norms = per_layer(c)["norms"] * c.L + c.d
    n = non_embedding(c)
    d_ff_exact = int(2 * (4 * c.d) / 3)
    mlp_exact = 3 * c.d * d_ff_exact
    n_exact = (attn + mlp_exact) * c.L + norms
    return {"ungated_4d_params": 2 * c.d * (4 * c.d),
            "gated_two_thirds_params": mlp_exact,
            "d_ff_two_thirds": d_ff_exact,
            "mlp_per_layer": mlp, "attn_per_layer": attn,
            "mlp_total": mlp * c.L, "attn_total": attn * c.L,
            "norm_total": norms, "non_embedding": n,
            "ffn_share": mlp * c.L / n,
            "non_embedding_two_thirds": n_exact,
            "saving": n - n_exact, "saving_frac": (n - n_exact) / n}


def report_ffn(c: Config = MODEL_D) -> None:
    b = ffn_budget(c)
    print(f"1  classical width 4d                {4*c.d:,}")
    print(f"   ungated params per layer 2 d 4d   {b['ungated_4d_params']:,}")
    print(f"2  two-thirds width                  {b['d_ff_two_thirds']:,}")
    print(f"   gated params per layer 3 d d_ff'  {b['gated_two_thirds_params']:,}"
          f"   ({100*abs(b['gated_two_thirds_params']-b['ungated_4d_params'])/b['ungated_4d_params']:.3f}% off)")
    p = llama_intermediate_size(c.d)
    print(f"3  Llama pipeline at d = {c.d}:"
          f"  4d = {p['4d']:,} -> x2/3 = {p['two_thirds']:,}"
          f" -> x1.3 = {p['multiplied']:,} -> up to 1024 = {p['intermediate_size']:,}")
    assert p["intermediate_size"] == c.d_ff, p
    q = llama_intermediate_size(2 * c.d)
    print(f"4  same pipeline at d = {2*c.d:,}"
          f"                     -> {q['intermediate_size']:,}  (the 70B check)")
    print(f"   ratio to the exact two-thirds value   {p['ratio_to_two_thirds']:.4f}")
    print(f"5  FFN per layer   {b['mlp_per_layer']:,}   x L = {b['mlp_total']:,}"
          f"  ({b['mlp_total']/1e9:.3f} B)")
    print(f"   attention/layer {b['attn_per_layer']:,}   x L = {b['attn_total']:,}"
          f"  ({b['attn_total']/1e9:.3f} B)")
    print(f"   norm gains                              {b['norm_total']:,}")
    print(f"   non-embedding                           {b['non_embedding']:,}"
          f"  ({b['non_embedding']/1e9:.3f} B)")
    print(f"   FFN share of non-embedding              {100*b['ffn_share']:.1f}%")
    print(f"   at the exact two-thirds width           {b['non_embedding_two_thirds']:,}"
          f"  ({b['non_embedding_two_thirds']/1e9:.3f} B)")
    print(f"   the 1.3x costs                          {b['saving']:,}"
          f"  ({100*b['saving_frac']:.1f}% of the model)")
    print(f"\n   activation memory, gated vs ungated at equal capacity:"
          f" 2 d_ff' / d_ff = {2*b['d_ff_two_thirds']/(4*c.d):.4f}")
    for s in (c.trained_context, c.extended_context):
        f_mlp = mlp_flops(c, 1)
        f_att = attention_flops(c, 1)["projections"] + 4 * s * c.d
        print(f"   s = {s:>7,}:  FFN {f_mlp/1e6:9.1f} MFLOP/token/layer"
              f"   attention {f_att/1e6:9.1f}   ratio {f_att/f_mlp:.3f}")
    s_star = (mlp_flops(c, 1) - attention_flops(c, 1)["projections"]) / (4 * c.d)
    print(f"   attention overtakes the FFN at s = {s_star:,.0f} = {s_star/c.d:.0f} d")


# ------------------------------------------- backward-pass activation memory
_BYTES = {"bf16": 2, "fp16": 2, "float16": 2, "bfloat16": 2, "fp32": 4, "float32": 4}


def activation_memory_backward(c: Config = MODEL_D, b: int = 1, s: int = 8192,
                               dtype: str = "bf16", fused_attention: bool = False
                               ) -> dict[str, object]:
    """Chapter 7's arithmetic box: what one block must keep for its backward pass.

    Every tensor the backward pass reads and cannot cheaply recompute, itemised.
    The RMS scalars are kept in fp32 whatever the activation dtype, because they
    are a reciprocal and the forward divides by them.

    fused_attention=True drops P, which a FlashAttention-style kernel recomputes
    from Q and K inside the backward rather than storing (§11.6).
    """
    w = _BYTES[dtype]
    wide = b * s * c.d * w                          # x, x_hat1, Q, O_cat, y, x_hat2
    kv = b * s * c.n_kv * c.d_h * w                 # K, V
    probs = b * c.h * s * s * w                     # P = softmax(S)
    ff = b * s * c.d_ff * w                         # G, U, A
    rms = 2 * b * s * 4                             # r1, r2, always fp32
    items = {"x, x_hat1, Q, O_cat, y, x_hat2": 6 * wide,
             "K, V": 2 * kv,
             "P = softmax(S)": 0 if fused_attention else probs,
             "G, U, A": 3 * ff,
             "r1, r2 (fp32)": rms}
    total = sum(items.values())
    return {"items": items, "total": total, "per_stack": total * c.L,
            "probs_share": (0.0 if fused_attention else probs / total),
            "boundary": wide, "probs": probs}


def checkpoint_memory(c: Config = MODEL_D, b: int = 1, s: int = 8192,
                      dtype: str = "bf16", fused_attention: bool = False
                      ) -> dict[str, float]:
    """M(m) = m*M_b + (L/m)*M_act, minimised over the number of segments m.

    The unconstrained optimum is the square-root rule, m* = sqrt(L*M_act/M_b).
    It is only reachable when m* <= L, i.e. when M_act/M_b <= L; past that the
    minimum sits on the boundary and the answer is 'checkpoint every layer'.
    Quoting the sqrt(L) rule without checking that condition is the trap.
    """
    import math
    a = activation_memory_backward(c, b, s, dtype, fused_attention)
    M_b, M_act = a["boundary"], a["total"]
    m_star = math.sqrt(c.L * M_act / M_b)
    clipped = m_star > c.L
    m = c.L if clipped else m_star
    M = lambda k: k * M_b + (c.L / k) * M_act
    return {"M_b": M_b, "M_act": M_act, "ratio": M_act / M_b,
            "m_star": m_star, "clipped": clipped, "m": m, "M": M(m),
            "M_sqrt_rule": 2 * math.sqrt(c.L * M_act * M_b),
            "unchecked": c.L * M_act}


def report_backward(c: Config = MODEL_D, b: int = 1, s: int = 8192,
                    dtype: str = "bf16") -> None:
    a = activation_memory_backward(c, b, s, dtype)
    print(f"one block's backward activation memory, b = {b}, s = {s:,}, {dtype}")
    for k, v in a["items"].items():
        print(f"  {k:<34} {v:>15,}  {v/1e6:>9.1f} MB")
    print(f"  {'TOTAL':<34} {a['total']:>15,}  {a['total']/1e9:>9.3f} GB"
          f"  ({a['total']/2**30:.2f} GiB)")
    print(f"  P is {100*a['probs_share']:.1f}% of the block;"
          f"  x L = {a['per_stack']/1e9:.1f} GB for the stack")
    f = activation_memory_backward(c, b, s, dtype, fused_attention=True)
    print(f"  with P recomputed:  {f['total']/1e9:.3f} GB per block,"
          f"  {f['per_stack']/1e9:.1f} GB for the stack")
    print("\ncheckpointing, M(m) = m M_b + (L/m) M_act")
    for fused in (False, True):
        r = checkpoint_memory(c, b, s, dtype, fused)
        tag = "fused  " if fused else "unfused"
        print(f"  {tag}  M_act/M_b = {r['ratio']:>5.1f}   m* = {r['m_star']:>5.1f}"
              f"   {'CLIPPED to m = L' if r['clipped'] else 'interior'}"
              f"   M = {r['M']/1e9:>6.3f} GB   (unchecked {r['unchecked']/1e9:.1f} GB)")


# ------------------------------------------------------- the loss, and units
# Chapter 10 (§10.5) refits L(N, D) = L_inf + A N^-alpha + B D^-beta.  Its five
# coefficients are frozen here because Chapter 8's arithmetic box derives its
# loss from them rather than quoting a training log.  If §10.5 moves any of
# them, Chapter 8's box re-runs; nothing in the prose is transcribed.
REFIT_2024 = {"L_inf": 1.82, "A": 482.0, "alpha": 0.348,
              "B": 2085.4, "beta": 0.366}
CHINCHILLA = {"L_inf": 1.70, "A": 406.4, "alpha": 0.34,
              "B": 410.7, "beta": 0.28}
BYTES_PER_TOKEN = 3.8          # measured; see figs/data/fig82_tokenizers.json
TRAINED_TOKENS = 15e12


def scaling_loss(N: float, D: float, fit: dict | None = None) -> dict[str, float]:
    """L(N, D) = L_inf + A N^-alpha + B D^-beta, itemised."""
    f = fit or REFIT_2024
    par = f["A"] * N ** -f["alpha"]
    dat = f["B"] * D ** -f["beta"]
    return {"L_inf": f["L_inf"], "parameter_term": par, "data_term": dat,
            "L": f["L_inf"] + par + dat,
            "floor_at_fixed_N": f["L_inf"] + par}


def loss_units(ce_nats: float, bytes_per_token: float = BYTES_PER_TOKEN,
               V: int = MODEL_D.V) -> dict[str, float]:
    """Chapter 8's four coordinates for one loss.  One measurement, four names."""
    import math
    bits = ce_nats / math.log(2)
    return {"nats_per_token": ce_nats, "bits_per_token": bits,
            "perplexity": math.exp(ce_nats),
            "bits_per_byte": bits / bytes_per_token,
            "uniform_nats": math.log(V), "uniform_ppl": float(V),
            "fraction_of_uniform": ce_nats / math.log(V)}


def tokens_for_loss(target: float, beta: float | None = None,
                    L_inf: float | None = None, D0: float = TRAINED_TOKENS,
                    L0: float | None = None) -> dict[str, float]:
    """How much data buys a target loss, along L(D) = L_inf + B D^-beta.

    The content is one line, equation (8.18):

        D2 / D1 = (Delta1 / Delta2) ** (1 / beta),   Delta = L - L_inf

    and everything Chapter 8's box prints is that with numbers substituted.
    Raises if the target is at or below the floor, because no quantity of data
    reaches it and returning a large number would hide that.
    """
    f = REFIT_2024
    beta = f["beta"] if beta is None else beta
    L_inf = f["L_inf"] if L_inf is None else L_inf
    if L0 is None:
        L0 = round(scaling_loss(non_embedding(MODEL_D), D0)["L"], 2)
    d1, d2 = L0 - L_inf, target - L_inf
    if d2 <= 0:
        raise ValueError(f"target {target} is at or below the floor {L_inf}; "
                         "no quantity of data reaches it")
    mult = (d1 / d2) ** (1.0 / beta)
    return {"from": L0, "target": target, "reducible_from": d1,
            "reducible_to": d2, "multiplier": mult, "tokens": D0 * mult,
            "exponent": 1.0 / beta}


def label_smoothing(eps: float, V: int = MODEL_D.V) -> dict[str, float]:
    """D-8.3: the logit gap the smoothed objective asks for, and its floor."""
    import math
    py, pj = 1 - eps + eps / V, eps / V
    return {"p_y": py, "p_j": pj, "ratio": py / pj,
            "gap": math.log(py / pj),
            "gap_large_V": math.log((1 - eps) * V / eps),
            "floor": -(py * math.log(py) + (V - 1) * pj * math.log(pj)),
            "floor_small_eps": eps * (1 + math.log(V / eps))}


def report_loss(c: Config = MODEL_D) -> None:
    import math
    N, D = non_embedding(c), TRAINED_TOKENS
    s = scaling_loss(N, D)
    print(f"the loss, from the 2024 refit at N = {N/1e9:.2f} B, D = {D/1e12:.0f} T")
    print(f"  L_inf                {s['L_inf']:.4f}")
    print(f"  A N^-alpha           {s['parameter_term']:.4f}")
    print(f"  B D^-beta            {s['data_term']:.4f}"
          f"   ({100*s['data_term']/s['L']:.1f}% of the printed loss)")
    print(f"  L                    {s['L']:.4f}  -> {s['L']:.2f} nats/token")
    print(f"  floor as D -> inf    {s['floor_at_fixed_N']:.4f}"
          f"   (so tokens alone can still buy {s['data_term']:.4f} nats, no more)")
    u = loss_units(round(s["L"], 2))
    print()
    print(f"  {'nats/token':<16}{u['nats_per_token']:>10.3f}")
    print(f"  {'bits/token':<16}{u['bits_per_token']:>10.3f}")
    print(f"  {'perplexity':<16}{u['perplexity']:>10.3f}")
    print(f"  {'bits/byte':<16}{u['bits_per_byte']:>10.3f}"
          f"   at {BYTES_PER_TOKEN} bytes/token")
    print(f"  {'ln V':<16}{u['uniform_nats']:>10.3f}   (PPL {u['uniform_ppl']:,.0f});"
          f" the loss is {100*u['fraction_of_uniform']:.0f}% of the way from 0 to it")
    print()
    print(f"  the next tenth, along L(D) = L_inf + B D^-beta"
          f"  (1/beta = {1/REFIT_2024['beta']:.3f})")
    for lab, tgt in (("literal step   2.03 -> 1.93", 1.93),
                     ("halve it       2.03 -> 1.925", 1.925)):
        r = tokens_for_loss(tgt)
        print(f"    {lab:<30} x{r['multiplier']:>6.2f}   {r['tokens']/1e12:>6.1f} T")
    prev = tokens_for_loss(2.03, L0=2.135)
    print(f"    {'previous tenth 2.135 -> 2.03':<30} x{prev['multiplier']:>6.2f}"
          f"   {prev['tokens']/1e12:>6.1f} T")
    for tgt in (2.03, 1.925):
        v = loss_units(tgt)
        print(f"    L {tgt:.3f} -> PPL {v['perplexity']:.3f}"
              f"  bits/token {v['bits_per_token']:.3f}"
              f"  bits/byte {v['bits_per_byte']:.3f}")
    ls = label_smoothing(0.1)
    print()
    print(f"  label smoothing at eps = 0.1, V = {c.V:,}:"
          f"  gap {ls['gap']:.4f} nats, floor {ls['floor']:.4f} nats")


# ---------------------------------------------------------- optimiser state
def params_by_ndim(c: Config = MODEL_D) -> dict[str, int]:
    """Chapter 9's split: what is a 2-D weight matrix and what is not.

    Muon's linear-minimisation oracle is about singular values, so it applies
    only where both axes are feature axes.  Embeddings index a vocabulary on
    one axis and gains are 1-D, so both stay on AdamW (D-9.4's failure mode).
    """
    gains = norm_stats(c)["rmsnorm_params"]
    two_d = non_embedding(c) - gains
    embed = (1 if c.tied else 2) * embedding(c)
    return {"two_d": two_d, "embeddings": embed, "gains": gains,
            "other": embed + gains, "total": two_d + embed + gains}


def optimiser_state(c: Config = MODEL_D, optimiser: str = "adamw",
                    moment_bytes: int = 4) -> dict[str, object]:
    """Bytes held per parameter, itemised, for one training step.

    `moment_bytes` is 4 for fp32 moments and 1 for the 8-bit variants.  The
    parameters and gradients are bf16 and the master copy is fp32 in every
    configuration, because §7.7 says an update below the format's unit roundoff
    is lost and the master copy is what prevents it.
    """
    p = params_by_ndim(c)
    n = p["total"]
    items = {"bf16 parameters": 2 * n, "bf16 gradients": 2 * n,
             "fp32 master weights": 4 * n}
    if optimiser == "adamw":
        items["first moment m"] = moment_bytes * n
        items["second moment v"] = moment_bytes * n
    elif optimiser == "muon":
        items["Muon momentum (2-D)"] = 4 * p["two_d"]
        items["AdamW m and v (rest)"] = 2 * moment_bytes * p["other"]
    else:
        raise ValueError(optimiser)
    total = sum(items.values())
    state = total - items["bf16 parameters"] - items["bf16 gradients"]
    return {"items": items, "total": total, "state": state,
            "bytes_per_param": total / n, "state_per_param": state / n,
            "n": n}


def report_optimiser(c: Config = MODEL_D) -> None:
    p = params_by_ndim(c)
    print(f"parameters: {p['total']:,} total")
    print(f"  2-D weight matrices (Muon-eligible)  {p['two_d']:,}")
    print(f"  embeddings + unembedding             {p['embeddings']:,}")
    print(f"  RMSNorm gains                        {p['gains']:,}")
    for opt in ("adamw", "muon"):
        r = optimiser_state(c, opt)
        print(f"\n{opt}:")
        for k, v in r["items"].items():
            print(f"  {k:<24} {v/1e9:>7.2f} GB")
        print(f"  {'TOTAL':<24} {r['total']/1e9:>7.2f} GB"
              f"   ({r['bytes_per_param']:.2f} B/param)")
        print(f"  {'state proper':<24} {r['state']/1e9:>7.2f} GB"
              f"   ({r['state_per_param']:.2f} B/param)")
        print(f"  {'on 8 devices, sharded':<24} {r['total']/8/1e9:>7.2f} GB each")
    a = optimiser_state(c, "adamw")["state"]
    m = optimiser_state(c, "muon")["state"]
    print(f"\nMuon saves {(a-m)/1e9:.1f} GB of state, a {100*(1-m/a):.0f}% reduction")
    q = optimiser_state(c, "adamw", moment_bytes=1)
    print(f"8-bit AdamW moments: {q['total']/1e9:.2f} GB"
          f"  ({q['total']/1e9-80:+.2f} GB against an 80 GB device)")


def adam_burst_bound(b1: float = 0.9, b2: float = 0.999) -> float:
    """Equation (9.21): the largest step one dominant gradient can take, in
    units of eta, after the second moment has decayed."""
    import math
    return (1 - b1) / math.sqrt(1 - b2)


def beta2_half_life(b2: float = 0.999) -> float:
    """Equation (9.22): how long a stale second moment takes to forget."""
    import math
    return math.log(2) / math.log(1 / b2)


def newton_schulz_poly(s, a: float = 3.4445, b: float = -4.7750,
                       c: float = 2.0315, steps: int = 1):
    """The odd polynomial of D-9.4 step 7, composed `steps` times.

    The coefficients are the most volatile item in Chapter 9 and live here so
    that the print edition quotes them once and a revision edits one line.
    """
    import numpy as np
    x = np.asarray(s, dtype=float)
    for _ in range(steps):
        x = a * x + b * x ** 3 + c * x ** 5
    return x


# --------------------------------------------------------------- norm sites
def norm_stats(c: Config = MODEL_D) -> dict[str, int]:
    """Chapter 5's arithmetic box.  2L + 1 sites: two per layer plus the final
    norm before the unembedding."""
    sites = 2 * c.L + 1
    return {"sites": sites,
            "rmsnorm_params": sites * c.d,
            "layernorm_params": sites * 2 * c.d,
            "qk_norm_params": 2 * c.d_h * c.L}


def report_norm_stats(c: Config = MODEL_D) -> None:
    s = norm_stats(c)
    n = non_embedding(c)
    print(f"norm sites (2L+1)     {s['sites']}")
    print(f"RMSNorm parameters    {s['rmsnorm_params']:,}"
          f"   ({100*s['rmsnorm_params']/n:.4f}% of N)")
    print(f"LayerNorm parameters  {s['layernorm_params']:,}")
    print(f"saving from dropping beta {s['rmsnorm_params']:,}"
          f"   ({100*s['rmsnorm_params']/n:.4f}% of N)")
    print(f"AdamW state at 12 B/param  RMS {12*s['rmsnorm_params']/1e6:.2f} MB"
          f" | LN {12*s['layernorm_params']/1e6:.2f} MB")
    print(f"QK-norm parameters    {s['qk_norm_params']:,}"
          f"   ({100*s['qk_norm_params']/n:.6f}% of N)")
    import math
    print(f"QK-norm logit bound   sqrt(d_h) = {math.sqrt(c.d_h):.4f}"
          f" ; max softmax ratio e^(2 sqrt d_h) = {math.exp(2*math.sqrt(c.d_h)):.3e}")
    print("reductions per token: RMSNorm 1 over d, LayerNorm 2 over d")


# ---------------------------------------------- Chapter 13: fine-tuning memory
FULL_FT_BYTES_PER_PARAM = 16      # bf16 weight 2 + bf16 grad 2 + fp32 master 4
                                  # + AdamW m 4 + v 4


def lora_params(c: Config = None, r: int = 16) -> dict[str, int]:
    """Adapter parameters for rank-r adapters on all four attention projections.

    An adapter on a d_in x d_out matrix costs r(d_in + d_out), not 2 r d_in.
    With grouped-query attention W_K and W_V are d x (n_kv d_h) and not d x d,
    so an accounting that assumes multi-head shapes over-counts.  That trap is
    E-13.7 and it is worth 23% on Model D.
    """
    c = c or MODEL_D
    q = r * (c.d + c.h * c.d_h)                 # W_Q
    o = r * (c.h * c.d_h + c.d)                 # W_O
    kv = 2 * r * (c.d + c.n_kv * c.d_h)         # W_K and W_V
    per_layer = q + o + kv
    mha = 4 * r * 2 * c.d                       # the same count assuming MHA
    return {"per_layer": per_layer, "total": per_layer * c.L,
            "mha_per_layer": mha, "mha_total": mha * c.L,
            "over_count": (mha - per_layer) * c.L}


def activation_bytes(c: Config = None, b: int = 8, s: int = 4096,
                     p_b: int = 2) -> int:
    """Checkpointed at layer boundaries: one saved tensor of shape (b, s, d)
    per layer.  This term is identical in all three regimes, which is the whole
    of design consequence 3."""
    c = c or MODEL_D
    return c.L * s * b * c.d * p_b


def finetune_memory(c: Config = None, r: int = 16, b: int = 8, s: int = 4096,
                    quant_bits: float = 4.127) -> dict[str, dict]:
    """A-13.1.  Three regimes, three terms each, and the smallest device.

    The three terms move independently and that is the point: the step from
    full fine-tuning to LoRA is optimiser state, the step from LoRA to QLoRA is
    weights, and activations never move at all.
    """
    c = c or MODEL_D
    n = total_params(c)
    lora = lora_params(c, r)["total"]
    act = activation_bytes(c, b, s)
    rows = {
        "full": {"weights": 0, "state": FULL_FT_BYTES_PER_PARAM * n, "act": act},
        "lora": {"weights": 2 * n, "state": FULL_FT_BYTES_PER_PARAM * lora,
                 "act": act},
        "qlora": {"weights": int(n * quant_bits / 8),
                  "state": FULL_FT_BYTES_PER_PARAM * lora, "act": act},
    }
    for v in rows.values():
        v["total"] = v["weights"] + v["state"] + v["act"]
    return rows


def smallest_device(total_bytes: float,
                    devices=(24e9, 40e9, 80e9, 141e9)) -> str:
    for d in devices:
        if total_bytes <= d:
            return f"1 x {d/1e9:.0f} GB"
    n = -(-total_bytes // 80e9)
    return f"{int(n)} x 80 GB"


def finetune_report(c: Config = None, r: int = 16, b: int = 8) -> None:
    c = c or MODEL_D
    n = total_params(c)
    lp = lora_params(c, r)
    print(f"Model D, {n/1e9:.2f} B parameters, rank {r} adapters, batch {b}\n")
    print(f"(a) full fine-tune: {FULL_FT_BYTES_PER_PARAM} B/param"
          f" -> {FULL_FT_BYTES_PER_PARAM*n/1e9:.1f} GB of state")
    print(f"(b) adapters on all four attention projections, with GQA:")
    print(f"      per layer {lp['per_layer']:,}   x {c.L} = {lp['total']/1e6:.2f} M"
          f"  ({100*lp['total']/n:.3f}% of the model)")
    print(f"      optimiser state {FULL_FT_BYTES_PER_PARAM*lp['total']/1e6:.0f} MB")
    print(f"      assuming MHA shapes instead: {lp['mha_total']/1e6:.2f} M,"
          f" {100*(lp['mha_total']/lp['total']-1):.0f}% high"
          f" ({lp['over_count']/1e6:.2f} M too many)")
    print()
    for s in (4096, 32768):
        rows = finetune_memory(c, r, b, s)
        print(f"  s = {s}:  activations {rows['full']['act']/1e9:.1f} GB")
        print(f"    {'':<12}{'weights':>10}{'grad+opt':>11}{'activations':>13}"
              f"{'total':>9}{'  smallest device':>20}")
        for key, nm in (("full", "full FT"), ("lora", f"LoRA r={r}"),
                        ("qlora", f"QLoRA r={r}")):
            v = rows[key]
            w = "(in state)" if key == "full" else f"{v['weights']/1e9:.2f} G"
            print(f"    {nm:<12}{w:>10}{v['state']/1e9:>10.2f}G"
                  f"{v['act']/1e9:>12.1f}G{v['total']/1e9:>8.1f}G"
                  f"{smallest_device(v['total']):>20}")
        print(f"    full/LoRA ratio {rows['full']['total']/rows['lora']['total']:.2f}"
              f"   LoRA/QLoRA {rows['lora']['total']/rows['qlora']['total']:.2f}")
        print()
    print("  the 137 -> 25 GB step is optimiser state, the 25 -> 13 GB step is")
    print("  weights, and 8.6 GB of activations never moves in either")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedding-share", action="store_true")
    ap.add_argument("--section", choices=["attention"])
    ap.add_argument("--rope-bands", action="store_true")
    ap.add_argument("--norm-stats", action="store_true")
    ap.add_argument("--ffn", action="store_true")
    ap.add_argument("--backward", action="store_true")
    ap.add_argument("--loss", action="store_true")
    ap.add_argument("--optimiser", action="store_true")
    ap.add_argument("--finetune-memory", dest="finetune", action="store_true")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("-s", type=int, default=None)
    a = ap.parse_args()
    if a.finetune:
        finetune_report()
    elif a.optimiser:
        report_optimiser()
    elif a.loss:
        report_loss()
    elif a.backward:
        report_backward(s=a.seq_len)
    elif a.ffn:
        report_ffn()
    elif a.norm_stats:
        report_norm_stats()
    elif a.rope_bands:
        report_rope_bands()
    elif a.embedding_share:
        report_embedding_share()
    elif a.section == "attention":
        report_attention(s=a.s)
    else:
        report_full()


if __name__ == "__main__":
    main()
