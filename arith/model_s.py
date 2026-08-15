"""Model S — the sparse mixture-of-experts reference model used by the Part III
arithmetic boxes.  DeepSeek-V3-shaped, defined by hyperparameters only.

The expert arithmetic below is exact and is what Chapters 3, 11 and 12 quote.
The attention block uses a documented MLA parameterisation; see NOTE.

    python arith/model_s.py             the parameter ledger
    python arith/model_s.py --moe       the Chapter 12 box
    python arith/model_s.py --capacity  expert capacity and dropped tokens
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    L: int = 61
    dense_layers: int = 3
    d: int = 7168
    h: int = 128
    d_h: int = 128
    d_c: int = 512            # MLA latent KV width
    d_r: int = 64             # decoupled RoPE width
    d_q: int = 1536           # compressed query width
    E: int = 256              # routed experts
    shared: int = 1
    k: int = 8                # experts active per token
    d_expert: int = 2048      # expert inner width
    d_ff_dense: int = 18432   # dense-layer FFN width
    V: int = 129280
    rope_base: int = 10_000
    trained_context: int = 4096
    extended_context: int = 131_072


MODEL_S = Config()


def expert_params(c: Config = MODEL_S) -> int:
    """One expert: a SwiGLU FFN of inner width d_expert."""
    return 3 * c.d * c.d_expert


def moe_layer_params(c: Config = MODEL_S) -> int:
    return (c.E + c.shared) * expert_params(c)


def moe_layer_active(c: Config = MODEL_S) -> int:
    return (c.k + c.shared) * expert_params(c)


def mla_params(c: Config = MODEL_S) -> int:
    """NOTE: an MLA parameterisation, not a transcription of any released
    checkpoint.  Down-projections to the latent, up-projections back to per-head
    K and V, a compressed query path, the decoupled RoPE dimensions, and W_O."""
    w_dkv = c.d * c.d_c
    w_uk = c.d_c * c.h * c.d_h
    w_uv = c.d_c * c.h * c.d_h
    w_q = c.d * c.d_q + c.d_q * c.h * c.d_h      # query is compressed too
    w_rope = c.d * c.d_r + c.d_c * c.h * c.d_r
    w_o = c.h * c.d_h * c.d
    return w_dkv + w_uk + w_uv + w_q + w_rope + w_o


def totals(c: Config = MODEL_S) -> dict[str, int]:
    n_moe = c.L - c.dense_layers
    dense_ffn = 3 * c.d * c.d_ff_dense
    attn = mla_params(c) * c.L
    norms = 2 * c.d * c.L
    total = (attn + norms
             + dense_ffn * c.dense_layers
             + moe_layer_params(c) * n_moe
             + 2 * c.V * c.d)
    active = (mla_params(c) * c.L + norms
              + dense_ffn * c.dense_layers
              + moe_layer_active(c) * n_moe
              + 2 * c.V * c.d)
    return {"expert": expert_params(c),
            "moe_layer": moe_layer_params(c),
            "moe_layer_active": moe_layer_active(c),
            "routed_total": c.E * expert_params(c) * n_moe,
            "total": total, "active": active}


# ------------------------------------------------------- Chapter 12 costs
BF16 = 2


def flops_per_token(n_active: int) -> int:
    """Chapter 10's 6ND, specialised.  Only the forward pass here, and only
    the parameters this token actually touches: 2 FLOPs per active parameter,
    one multiply and one add."""
    return 2 * n_active


def weight_bytes(n_total: int, p_b: int = BF16) -> int:
    """Every parameter is resident whether or not the token routes to it.  That
    asymmetry with flops_per_token is the whole of M-12.1."""
    return n_total * p_b


def devices(n_total: int, capacity_bytes: float, p_b: int = BF16) -> int:
    """Weights only.  A real deployment also needs activations, the KV cache of
    Chapter 11, and room for the optimiser if it is training, so this is a
    floor on the device count and not an estimate of one."""
    import math
    return math.ceil(weight_bytes(n_total, p_b) / capacity_bytes)


def expert_capacity(T: int, E: int, k: int, c: float) -> dict[str, float]:
    """Section 12.3.  The buffer each expert is given, per micro-batch.

    The k is the part secondary sources drop.  With top-k routing there are
    kT assignments to distribute over E experts, not T, so the mean load is
    kT/E and the buffer is c times that.  Omitting k under-sizes the buffer by
    a factor of k, which for Model S is eight.
    """
    mean = k * T / E
    return {"mean_load": mean, "capacity": c * mean,
            "slack_fraction": 1.0 - 1.0 / c if c > 0 else 0.0,
            "wasted_slots": (c - 1.0) * mean * E}


def dropped_fraction(loads, capacity: float) -> float:
    """A token over capacity is not an error.  It skips the expert and passes
    down the residual stream unchanged, so the layer becomes an identity map
    for that token: no exception, no NaN, just a hole."""
    over = sum(max(0.0, float(x) - capacity) for x in loads)
    return over / float(sum(loads))


def moe_report(c: Config = MODEL_S) -> None:
    try:
        from arith.model_d import MODEL_D, non_embedding, total_params
    except ImportError:                            # run as a script from arith/
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from arith.model_d import MODEL_D, non_embedding, total_params
    t = totals(c)
    n_moe = c.L - c.dense_layers
    ffn_active = moe_layer_active(c) * n_moe
    print(f"one routed expert, SwiGLU at inner width {c.d_expert}, d = {c.d}")
    print(f"  3 x {c.d} x {c.d_expert} = {expert_params(c):,} = {expert_params(c)/1e6:.2f} M")
    print(f"per MoE layer, {c.E} routed + {c.shared} shared")
    print(f"  {c.E + c.shared} x {expert_params(c)/1e6:.2f} M = {moe_layer_params(c)/1e9:.4f} B")
    print(f"  x {n_moe} MoE layers      = {moe_layer_params(c)*n_moe/1e9:.2f} B")
    print(f"total                    = {t['total']/1e9:.3f} B")
    print(f"active FFN, (k={c.k} + {c.shared}) x {n_moe} layers = {ffn_active/1e9:.3f} B")
    print(f"active elsewhere (MLA, dense FFN, embeddings) = "
          f"{(t['active']-ffn_active)/1e9:.3f} B")
    print(f"active per token         = {t['active']/1e9:.3f} B"
          f"   ({100*t['active']/t['total']:.1f}% of total)")
    print()
    fl, mem = flops_per_token(t["active"]), weight_bytes(t["total"])
    dense = flops_per_token(t["total"])
    print(f"forward FLOPs/token      {fl/1e9:.2f} GFLOP")
    print(f"  a dense model of the same total size would need {dense/1e9:.1f} GFLOP,"
          f" {dense/fl:.2f}x more")
    print(f"bf16 weight memory       {mem/1e9:.1f} GB")
    print(f"  the same number twice: 2 x {t['total']/1e9:.1f}e9 is the dense FLOP count"
          f" and the byte count, read in different units")
    for cap in (80e9, 141e9):
        print(f"  devices at {cap/1e9:.0f} GB, weights only: {devices(t['total'], cap)}")
    d = MODEL_D
    print()
    print(f"against Model D ({non_embedding(d)/1e9:.2f} B non-embedding):")
    print(f"  compute  {fl/1e9:.1f} against {flops_per_token(non_embedding(d))/1e9:.1f}"
          f" GFLOP/token = {fl/flops_per_token(non_embedding(d)):.2f}x")
    print(f"  memory   {mem/1e9:.0f} against {weight_bytes(total_params(d))/1e9:.1f}"
          f" GB = {mem/weight_bytes(total_params(d)):.1f}x")
    print("  cheap in the dimension that sets throughput, expensive in the one")
    print("  that decides whether you can run it at all")


def capacity_report(c: Config = MODEL_S, T: int = 8192) -> None:
    print(f"T = {T} tokens per micro-batch, E = {c.E} experts, k = {c.k}\n")
    print(f"{'c':>6}{'capacity':>12}{'mean load':>12}{'slack':>9}{'wasted slots':>15}")
    for cf in (1.0, 1.25, 1.5, 2.0):
        r = expert_capacity(T, c.E, c.k, cf)
        print(f"{cf:>6.2f}{r['capacity']:>12.0f}{r['mean_load']:>12.0f}"
              f"{100*r['slack_fraction']:>8.0f}%{r['wasted_slots']:>15,.0f}")
    print(f"\n  without the k the capacity would be c T / E = {T/c.E:.0f} at c = 1,")
    print(f"  which is {c.k}x too small: there are k T = {c.k*T:,} assignments, not T")


def report(c: Config = MODEL_S) -> None:
    t = totals(c)
    print(f"one expert            {t['expert']:,}   ({t['expert']/1e6:.2f} M)")
    print(f"per MoE layer, all    {t['moe_layer']:,}")
    print(f"per MoE layer, active {t['moe_layer_active']:,}   ({t['moe_layer_active']/1e6:.1f} M)")
    print(f"routed experts, all   {t['routed_total']/1e9:.1f} B")
    print(f"MLA per layer         {mla_params(c):,}")
    print(f"total                 {t['total']/1e9:.0f} B")
    print(f"active per token      {t['active']/1e9:.0f} B")
    print(f"active fraction       {100*t['active']/t['total']:.1f}%")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--moe", action="store_true", help="the Chapter 12 box")
    ap.add_argument("--capacity", action="store_true", help="section 12.3")
    a = ap.parse_args()
    if a.moe:
        moe_report()
    elif a.capacity:
        capacity_report()
    else:
        report()


if __name__ == "__main__":
    main()
