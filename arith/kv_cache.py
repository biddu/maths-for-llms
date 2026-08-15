"""The KV cache, and everything Chapter 11's arithmetic box prints.

    python arith/kv_cache.py                the A-11.1 table
    python arith/kv_cache.py --capacity     what fits on a 141 GB part
    python arith/kv_cache.py --intensity    decode arithmetic intensity, and the roofline
    python arith/kv_cache.py --linear       the linear-attention crossovers of section 11.8

Two shapes of cache appear in the book and they are not the same formula.

A *per-head* cache stores K and V for every key/value head: 2 tensors, n_kv
heads, d_h wide.  A *latent* cache stores one compressed vector per token plus
a decoupled positional part shared across heads: d_c + d_r elements, with no
factor of two, because K and V are both reconstructed from the same latent.
Writing the second as if it were the first is the commonest error in
reproductions of these numbers, and it is off by 2 n_kv d_h / (d_c + d_r).
"""
from __future__ import annotations
import argparse

try:
    from arith.model_d import MODEL_D, total_params
    from arith.model_s import MODEL_S, totals
    from arith.accelerators import DEFAULT, GiB
except ImportError:                                   # run as a script from arith/
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from arith.model_d import MODEL_D, total_params
    from arith.model_s import MODEL_S, totals
    from arith.accelerators import DEFAULT, GiB

KiB = 1 << 10
MiB = 1 << 20

BF16, FP8, FP32 = 2, 1, 4


# ------------------------------------------------------------------ bytes
def per_head_bytes_per_token(L: int, n_kv: int, d_h: int, p_b: int = BF16) -> int:
    """(11.2) at s = b = 1.

    The leading 2 is K-and-V, two tensors.  It is *not* bytes per element:
    that is p_b, which enters as a separate factor and happens also to equal 2
    in bf16.  The collision is the reason this function spells both out.
    """
    return 2 * L * n_kv * d_h * p_b


def latent_bytes_per_token(L: int, d_c: int, d_r: int, p_b: int = BF16) -> int:
    """(11.18).  One latent c_j of width d_c per token per layer, plus one
    decoupled positional vector of width d_r shared across all heads.  No
    factor of two: K and V are both reconstructed from c_j."""
    return (d_c + d_r) * L * p_b


def cache_bytes(bytes_per_token: int, s: int, b: int = 1) -> int:
    return bytes_per_token * s * b


def schemes(p_b: int = BF16) -> dict[str, dict]:
    """Every row of A-11.1.  The three Model D rows differ only in n_kv, which
    is the point of the ladder: MQA, GQA and MHA are one formula at three
    values of a single integer."""
    d, s_ = MODEL_D, MODEL_S
    return {
        "D MHA": {"model": "D", "n_kv": d.h, "params": total_params(d),
                  "bytes": per_head_bytes_per_token(d.L, d.h, d.d_h, p_b)},
        "D GQA": {"model": "D", "n_kv": d.n_kv, "params": total_params(d),
                  "bytes": per_head_bytes_per_token(d.L, d.n_kv, d.d_h, p_b)},
        "D MQA": {"model": "D", "n_kv": 1, "params": total_params(d),
                  "bytes": per_head_bytes_per_token(d.L, 1, d.d_h, p_b)},
        "S MHA": {"model": "S", "n_kv": s_.h, "params": totals(s_)["total"],
                  "bytes": per_head_bytes_per_token(s_.L, s_.h, s_.d_h, p_b)},
        "S MLA": {"model": "S", "n_kv": None, "params": totals(s_)["total"],
                  "bytes": latent_bytes_per_token(s_.L, s_.d_c, s_.d_r, p_b)},
    }


# ------------------------------------------------------------- intensity
def decode_intensity(h: int, n_kv: int, p_b: int = BF16) -> float:
    """(11.5).  I = 2h/(n_kv p_b) FLOP per byte, independent of s, L and d_h.

    Numerator 4 L h d_h s FLOPs, denominator 2 L n_kv d_h s p_b bytes; L, d_h
    and s cancel identically.  What survives is the ratio of query heads to
    key/value heads, and the precision of the cache.  Nothing else in the
    architecture, and nothing at all about the context length.
    """
    return 2.0 * h / (n_kv * p_b)


def decode_flops(L: int, h: int, d_h: int, s: int) -> int:
    """One decode step's attention FLOPs: q K^T then the value mix, both
    2 d_h s per head, over h heads and L layers."""
    return 4 * L * h * d_h * s


def latency_floor(bytes_moved: int, bandwidth: float = None) -> float:
    """Seconds per token if the only traffic were the cache.  A lower bound on
    decode latency, and one that no kernel work can improve."""
    return bytes_moved / (bandwidth or DEFAULT.bandwidth)


def concurrency(bytes_per_token: int, s: int, weight_bytes: int,
                hbm: int = None) -> dict[str, float]:
    """How many sequences of length s the scheduler may admit.

    The floor matters: admitting the fractional part means the scheduler
    preempts and re-prefills, and the symptom is p99 latency degrading while
    utilisation falls.
    """
    hbm = hbm or DEFAULT.hbm_bytes
    free = hbm - weight_bytes
    per_seq = cache_bytes(bytes_per_token, s)
    return {"free_gib": free / GiB, "per_seq_gib": per_seq / GiB,
            "exact": free / per_seq, "admit": int(free // per_seq)}


# --------------------------------------------------- section 11.8 costs
def linear_attention(m: int, c=None) -> dict[str, float]:
    """Section 11.8's accounting, with the feature dimension m left free.

    Per layer, softmax attention costs 4 s^2 d FLOPs (two matmuls, each
    2 s^2 d) and caches 2 n_kv d_h s elements.  A factorised kernel with
    feature map into R^m costs 4 s m d and carries a state of m d elements:
    h heads each holding an m x d_h numerator, and h d_h = d.

    Two crossovers follow, and they are different questions with different
    answers.  Work: s = m, independent of d.  State against cache:
    s = m d / (2 n_kv d_h), which for Model D is 2m.
    """
    c = c or MODEL_D
    state = m * c.d                                     # elements per layer
    return {"m": m,
            "flop_crossover": float(m),
            "state_elements_per_layer": state,
            "state_bytes": state * c.L * BF16,
            "state_equals_cache_at": m * c.d / (2 * c.n_kv * c.d_h)}


def dsa_breakeven(k: int, h_i: int, d_i: int, c=None) -> dict[str, float]:
    """Top-k selection with a cheap indexer, against full attention.

    Full attention costs 4 s^2 d per layer.  Selection costs 4 s k d for the
    attention itself plus 2 s^2 h_i d_i for the indexer, which still scores
    every pair; only the value mix is avoided.  Setting the two equal:

        s = 2 k d / (2 d - h_i d_i)

    and the denominator carries the whole story.  An indexer of total width
    h_i d_i >= 2d never pays for itself at any context length, because
    scoring with the surrogate then costs as much as scoring with the real
    thing.  Cheap has a precise meaning here and it is h_i d_i << 2d.
    """
    c = c or MODEL_D
    den = 2 * c.d - h_i * d_i
    return {"k": k, "indexer_width": h_i * d_i, "budget": 2 * c.d,
            "breakeven_s": (2 * k * c.d / den) if den > 0 else float("inf")}


def window_horizon(w: int, L: int) -> int:
    """A sliding window of width w admits w positions per layer; stacking L of
    them gives a horizon of L w.  Reaching the far end takes L hops of mixing,
    which is why the horizon and the trained context being equal is a weaker
    statement than it sounds."""
    return w * L


# ---------------------------------------------------------------- report
def table(p_b: int = BF16, lengths=(8192, 32768, 131072)) -> None:
    rows = schemes(p_b)
    head = f"{'model':<7}{'scheme':<8}{'params':>10}{'bytes/token':>14}"
    head += "".join(f"{'s = ' + f'{s:,}':>14}" for s in lengths)
    print(head)
    for name, r in rows.items():
        model, scheme = name.split()
        unit = f"{r['bytes']/KiB:.4g} KiB" if r["bytes"] < MiB else f"{r['bytes']/MiB:.4f} MiB"
        line = f"{model:<7}{scheme:<8}{r['params']/1e9:>9.2f}B{unit:>14}"
        line += "".join(f"{cache_bytes(r['bytes'], s)/GiB:>13.4g}G" for s in lengths)
        print(line)
    d_mha, d_gqa, s_mla = rows["D MHA"]["bytes"], rows["D GQA"]["bytes"], rows["S MLA"]["bytes"]
    print(f"\n  S MLA is {rows['S MHA']['bytes']/s_mla:.2f}x smaller than the same model with MHA,")
    print(f"  {d_mha/s_mla:.2f}x smaller than an 8B model with MHA,"
          f" and {d_gqa/s_mla:.2f}x smaller than the same 8B with GQA.")
    print("  Cache size is set by architecture, not by parameter count.")


def report(which: str = "table") -> None:
    d, s_ = MODEL_D, MODEL_S
    if which == "table":
        table()
        return
    if which == "capacity":
        w = 2 * total_params(d)
        print(f"Model D weights, bf16: {w/GiB:.3f} GiB on a {DEFAULT.capacity_gib:.1f} GiB part")
        for s in (8192, 131072):
            c = concurrency(schemes()["D GQA"]["bytes"], s, w)
            print(f"  s = {s:>7,}: {c['per_seq_gib']:>8.4g} GiB per sequence,"
                  f" {c['free_gib']:.2f} free -> {c['exact']:.4f}, admit {c['admit']}")
        return
    if which == "intensity":
        print(f"machine balance {DEFAULT.balance:.1f} FLOP/byte\n")
        print(f"{'scheme':<8}{'n_kv':>6}{'bf16':>10}{'fp8':>10}{'below ridge (bf16)':>22}")
        for name, n_kv in (("MHA", d.h), ("GQA", d.n_kv), ("MQA", 1)):
            i2, i1 = decode_intensity(d.h, n_kv, BF16), decode_intensity(d.h, n_kv, FP8)
            print(f"{name:<8}{n_kv:>6}{i2:>9.0f}F/B{i1:>9.0f}F/B{DEFAULT.balance/i2:>20.1f}x")
        b = cache_bytes(schemes()["D GQA"]["bytes"], 131072)
        t = latency_floor(b)
        print(f"\nreading {b/GiB:.0f} GiB at {DEFAULT.bandwidth/1e12:.2f} TB/s takes"
              f" {1e3*t:.4f} ms -> at most {1/t:.1f} tokens/s at b = 1")
        print("  before a single weight byte is touched")
        return
    if which == "linear":
        print("linear attention, per layer, feature dimension m free:")
        print(f"{'m':>7}  {'named as':<26}{'state':>12}{'FLOP xover':>12}{'state = cache at':>18}")
        for m, note in ((d.d_h, "d_h, elu+1"), (621, "d_h log d_h, Performer"),
                        (d.d, "d, the widest reading")):
            r = linear_attention(m)
            print(f"{m:>7}  {note:<26}{r['state_bytes']/GiB:>10.4f}G"
                  f"{r['flop_crossover']:>12.0f}{r['state_equals_cache_at']:>18.0f}")
        print(f"\ntop-k selection at k = 2048, against 2d = {2*d.d}:")
        for h_i in (4, 8, 16, 64):
            r = dsa_breakeven(2048, h_i, 128)
            be = f"{r['breakeven_s']:.0f}" if r["breakeven_s"] < 1e12 else "never"
            print(f"  indexer {h_i:>3} heads x 128 = width {r['indexer_width']:>5}"
                  f"  -> break-even s = {be}")
        print(f"\nsliding window w = 4096 over L = {d.L}: horizon"
              f" {window_horizon(4096, d.L):,}")
        return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ("capacity", "intensity", "linear"):
        ap.add_argument(f"--{flag}", action="store_true")
    a = ap.parse_args()
    report("capacity" if a.capacity else "intensity" if a.intensity
           else "linear" if a.linear else "table")


if __name__ == "__main__":
    main()
