"""Machine balance, in one file, because it is the fastest-ageing number in the book.

Chapter 11 derives one hardware-independent quantity, the arithmetic intensity
of decode, I = 2h/(n_kv p_b), and then compares it to *one* hardware number: the
ratio of peak arithmetic rate to memory bandwidth.  That ratio is the ridge
point of the roofline model, and it is the only place in Part III where a
current accelerator's specification enters the argument.

Keeping it here means a second edition edits this file.  Nothing else in the
book quotes a bandwidth or a TFLOP/s figure.

    python arith/accelerators.py       the table, and Model D's position on it
"""
from __future__ import annotations
from dataclasses import dataclass

GiB = 1 << 30
TiB = 1 << 40


@dataclass(frozen=True)
class Accelerator:
    name: str
    hbm_bytes: int            # marketing capacity, decimal, as sold
    bandwidth: float          # bytes/second, peak
    flops_bf16: float         # dense bf16 FLOP/s, no sparsity
    sram_per_sm: int          # bytes of SRAM per streaming multiprocessor
    sms: int

    @property
    def balance(self) -> float:
        """The ridge point: FLOPs per byte at which compute and memory are in
        balance.  Below it a kernel is bandwidth-bound and its FLOP count is
        irrelevant; above it the reverse."""
        return self.flops_bf16 / self.bandwidth

    @property
    def capacity_gib(self) -> float:
        return self.hbm_bytes / GiB


# The part Chapter 11's arithmetic box is sized against.  141 GB is the
# marketing figure, decimal, which is 131.3 GiB of addressable memory: the
# distinction is worth 10 GiB and every capacity plan that ignores it is wrong
# by about one 128k sequence.
H200 = Accelerator("141 GB part", 141_000_000_000, 3.35e12, 989e12, 228 << 10, 132)
A100 = Accelerator("80 GB part", 80_000_000_000, 2.039e12, 312e12, 192 << 10, 108)

DEFAULT = H200


def sram_elements(a: Accelerator = DEFAULT, p_b: int = 2) -> int:
    """M in FlashAttention's IO bound, in elements rather than bytes."""
    return a.sram_per_sm // p_b


def io_advantage(d_h: int, a: Accelerator = DEFAULT, p_b: int = 2) -> float:
    """Standard attention moves Theta(s^2) elements; the tiled algorithm moves
    Theta(s^2 d_h^2 / M).  The ratio is M/d_h^2, and it is a modest constant
    rather than an asymptotic win: the exponent on s does not change."""
    return sram_elements(a, p_b) / d_h ** 2


def report() -> None:
    try:
        from arith.model_d import MODEL_D
    except ImportError:                               # run as a script from arith/
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from arith.model_d import MODEL_D
    print(f"{'part':<14}{'capacity':>12}{'bandwidth':>14}{'bf16':>14}{'balance':>12}")
    for a in (H200, A100):
        print(f"{a.name:<14}{a.capacity_gib:>10.1f}GiB{a.bandwidth/1e12:>11.3f}TB/s"
              f"{a.flops_bf16/1e12:>11.0f}TF/s{a.balance:>10.1f} F/B")
    d = MODEL_D
    intensity = 2 * d.h / (d.n_kv * 2)
    print(f"\nModel D decode attention, bf16 GQA: {intensity:.0f} FLOP/byte")
    print(f"  that is {DEFAULT.balance / intensity:.1f}x below the ridge, so decode time is"
          f" bytes/bandwidth and the FLOP count never enters")
    print(f"\ntiled attention moves M/d_h^2 = {io_advantage(d.d_h):.2f}x fewer HBM"
          f" elements at d_h = {d.d_h}")
    print(f"  (M = {sram_elements()} bf16 elements of SRAM per SM)")


if __name__ == "__main__":
    report()
