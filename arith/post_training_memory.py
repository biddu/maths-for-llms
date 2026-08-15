"""What each post-training algorithm costs, in memory and in compute.

    python arith/post_training_memory.py              the A-15 table
    python arith/post_training_memory.py --70b        E-15.8
    python arith/post_training_memory.py --compute    E-15.9, the GRPO trade
    python arith/post_training_memory.py --rlvr       Corollary 15.1

One idea runs through Chapter 15: every named algorithm is PPO with one of its
four resident models deleted.  PPO holds a trainable policy, a frozen reference,
a frozen reward model and a trainable value network.  DPO deletes the reward
model and the value network by making the reward implicit; GRPO deletes the
value network by replacing it with a group baseline; RLVR deletes the reward
model in favour of a subprocess.  This file counts what each deletion is worth.

The per-parameter costs are Chapter 9's, byte for byte, and Chapter 13's:

    trainable, mixed-precision AdamW   16 B  (bf16 w 2, bf16 grad 2,
                                              fp32 master 4, m 4, v 4)
    frozen, bf16                        2 B

Note that the totals here are *state* only.  Activations and the rollout KV
cache (Chapter 11) are on top, and for PPO they are what turns 3.6 devices into
eight.
"""
from __future__ import annotations

import argparse
import math

TRAINABLE_BYTES = 16
FROZEN_BYTES = 2
GB = 1e9


def _model_d_params() -> int:
    try:
        from arith.model_d import MODEL_D, total_params
    except ImportError:
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from arith.model_d import MODEL_D, total_params
    return total_params(MODEL_D)


# Which of the four models each algorithm keeps resident.  This table IS the
# chapter: read down a column and you are reading a section heading.
RESIDENT = {
    "PPO":   {"policy": True, "reference": False, "reward": False, "value": True},
    "DPO":   {"policy": True, "reference": False},
    "GRPO":  {"policy": True, "reference": False, "reward": False},
    "GRPO+verifier": {"policy": True, "reference": False},
    "RLVR":  {"policy": True, "reference": False},
}


def state_bytes(n_params: int, algorithm: str = "PPO") -> dict[str, float]:
    """Bytes of weights and optimiser state, by component.

    `RESIDENT[a][component]` is True when the component is trainable, so it
    carries optimiser state, and False when it is frozen at inference precision.
    """
    out = {}
    for name, trainable in RESIDENT[algorithm].items():
        out[name] = n_params * (TRAINABLE_BYTES if trainable else FROZEN_BYTES)
    out["total"] = sum(out.values())
    return out


def devices(total_bytes: float, per_device_gb: float = 80.0,
            activation_headroom: float = 0.0) -> int:
    """Smallest device count, with a fraction of each device reserved.

    `activation_headroom=0.3` reserves 30% for activations and the rollout KV
    cache, which is E-15.8's setting and is the honest one: a state count that
    exactly fills a device does not run.
    """
    usable = per_device_gb * GB * (1.0 - activation_headroom)
    return int(math.ceil(total_bytes / usable))


# ------------------------------------------------------------------ compute
# Forward-equivalents per prompt, in units of N*L, with 2N FLOP per token
# forward and 4N per token backward.  Generation is a forward pass per token.
FWD, BWD = 2, 4


def compute_per_prompt(algorithm: str, G: int = 8, verifier: bool = False) -> int:
    """PPO pays for one rollout and two trainable models; GRPO pays for G
    rollouts and one.  The units cancel, so only the ratio matters."""
    if algorithm == "PPO":
        return (FWD                      # rollout
                + FWD + BWD              # policy forward and backward
                + FWD + BWD              # value network, same size
                + FWD                    # reference
                + FWD)                   # reward model
    if algorithm == "GRPO":
        per = FWD + (FWD + BWD) + FWD + (0 if verifier else FWD)
        return per * G
    if algorithm == "DPO":
        return (FWD + BWD) + FWD         # policy, and the frozen reference
    raise KeyError(algorithm)


def grpo_crossover(verifier: bool = False) -> int:
    """The G at which GRPO's compute per prompt first exceeds PPO's.  It is
    small, which is the point: GRPO trades a large memory saving for a compute
    cost that arrives almost immediately."""
    ppo = compute_per_prompt("PPO")
    return next(G for G in range(1, 1000)
                if compute_per_prompt("GRPO", G, verifier) > ppo)


# ------------------------------------------------------------- Corollary 15.1
def rlvr_logit_shift(p: float, beta: float) -> float:
    """(15.22).  pi*(C|x) from a pass rate p and a KL coefficient beta.

    The whole content is that logit pi*(C) = logit p + 1/beta: the shift is the
    same for every prompt whatever its difficulty, and p = 0 stays p = 0.
    """
    e = math.exp(1.0 / beta)
    return p * e / (p * e + 1.0 - p)


def beta_for_target(p: float, target: float) -> float:
    """The beta that moves a pass rate p to a target, by inverting (15.22)."""
    logit = lambda q: math.log(q / (1 - q))
    return 1.0 / (logit(target) - logit(p))


def rollouts_for_signal(p: float, confidence: float = 0.99) -> int:
    """k samples contain at least one success with probability 1 - (1-p)^k.
    This is where Chapter 10's marginal compute moves into the rollout budget."""
    if p <= 0:
        return -1
    return int(math.ceil(math.log(1 - confidence) / math.log(1 - p)))


# ---------------------------------------------------------------- reporting
def report(which: str = "table") -> None:
    n = _model_d_params()
    if which == "70b":
        n = int(70e9)
        print(f"E-15.8  a {n/1e9:.0f} B dense model, 30% reserved for activations")
        print(f"  {'algorithm':<16}{'state, GB':>12}{'80 GB':>9}{'141 GB':>9}")
        for a in ("PPO", "GRPO", "DPO"):
            t = state_bytes(n, a)["total"]
            print(f"  {a:<16}{t/GB:>12.0f}"
                  f"{devices(t, 80, 0.3):>9}{devices(t, 141, 0.3):>9}")
        return
    if which == "compute":
        ppo = compute_per_prompt("PPO")
        print("E-15.9  forward-equivalents per prompt, in units of N L\n")
        print(f"  PPO = {ppo}\n")
        print(f"  {'G':>4}{'GRPO, reward model':>22}{'GRPO, verifier':>17}"
              f"{'vs PPO':>10}")
        for G in (1, 2, 4, 8, 16, 32):
            a, b = compute_per_prompt("GRPO", G), compute_per_prompt("GRPO", G, True)
            print(f"  {G:>4}{a:>22}{b:>17}{a/ppo:>9.2f}x")
        print(f"\n  crossover at G = {grpo_crossover()} with a reward model, "
              f"G = {grpo_crossover(True)} with a verifier.")
        saved = _model_d_params() * TRAINABLE_BYTES / GB
        print(f"  At G = 8 that is {compute_per_prompt('GRPO', 8)/ppo:.1f}x PPO's "
              f"compute to save the value network's {saved:.0f} GB.")
        print("  GRPO is a memory-for-compute trade and the exchange rate is steep.")
        return
    if which == "rlvr":
        print("Corollary 15.1  RLVR shifts the log-odds of being correct\n")
        print(f"  {'pass rate p':>12}{'beta':>8}{'pi*(C|x)':>12}"
              f"{'rollouts for 99%':>18}")
        for p in (0.0, 0.01, 0.05, 0.12, 0.3, 0.6):
            for beta in (0.5019,):
                k = rollouts_for_signal(p)
                print(f"  {p:>12}{beta:>8}{rlvr_logit_shift(p, beta):>12.6f}"
                      f"{(k if k > 0 else 'never'):>18}")
        print(f"\n  E-15.10: beta for p = 0.12 -> 0.5 is "
              f"{beta_for_target(0.12, 0.5):.6f},")
        print(f"  and {rollouts_for_signal(0.12)} rollouts give a success with "
              f"probability {1-(1-0.12)**rollouts_for_signal(0.12):.4f}.")
        print("  Note the first row: no beta buys signal where the reference")
        print("  never succeeds, which is D-15.3's absolute-continuity clause.")
        return

    print(f"A-15  post-training state on Model D, {n:,} parameters\n")
    print(f"  {'component':<28}{'PPO':>10}{'DPO':>10}{'GRPO':>10}")
    comps = ("policy", "reference", "reward", "value")
    labels = {"policy": "policy (trainable)",
              "reference": "reference (frozen)",
              "reward": "reward model (frozen)",
              "value": "value network (trainable)"}
    for c in comps:
        row = ""
        for a in ("PPO", "DPO", "GRPO"):
            b = state_bytes(n, a)
            row += f"{b[c]/GB:>10.2f}" if c in b else f"{'--':>10}"
        print(f"  {labels[c]:<28}{row}")
    row = "".join(f"{state_bytes(n, a)['total']/GB:>10.2f}"
                  for a in ("PPO", "DPO", "GRPO"))
    print(f"  {'TOTAL, GB':<28}{row}")
    row = "".join(f"{state_bytes(n, a)['total']/GB/80:>10.2f}"
                  for a in ("PPO", "DPO", "GRPO"))
    print(f"  {'devices of 80 GB, state only':<28}{row}")
    print(f"\n  with a verifier instead of a reward model, GRPO is "
          f"{state_bytes(n, 'GRPO+verifier')['total']/GB:.2f} GB.")
    print(f"  On a 4 x 80 GB node (320 GB) PPO does not fit and the others do.")
    print(f"  PPO is exactly {state_bytes(n,'PPO')['total']/state_bytes(n,'DPO')['total']:.3f}x "
          f"DPO's state, because the value network is the same size as the policy.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--70b", dest="b70", action="store_true")
    ap.add_argument("--compute", action="store_true")
    ap.add_argument("--rlvr", action="store_true")
    a = ap.parse_args()
    report("70b" if a.b70 else "compute" if a.compute else
           "rlvr" if a.rlvr else "table")


if __name__ == "__main__":
    main()
