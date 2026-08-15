"""Checkpoint measurements for Maths for LLMs.

Three figures in this book depend on statistics of a trained model rather than
on arithmetic, and none of them can be written until the measurement exists:

    Chapter 2, F-2.4   raw against mean-centred cosine on the embedding matrix
    Chapter 3, F-3.6   where the attention mass goes, and the position-0 share
    Chapter 5, F-5.1   |mean(x)| / RMS(x) of the residual stream, by layer

This script produces all three from one checkpoint, writes the derived
statistics to measure/data/ as small arrays, and writes a manifest recording
exactly what was measured.  The checkpoint itself is never committed; the
statistics and the manifest are, so the figures rebuild on a fresh clone and in
CI without a download.

    python measure/checkpoint_stats.py --model <hf-id> --revision <sha>
    python measure/checkpoint_stats.py --model <hf-id> --what embeddings

Design notes worth knowing before you run it:

* **Attention weights need eager attention.**  SDPA and FlashAttention never
  materialise the s x s matrix, which is the whole point of them (§11.6), so
  `output_attentions=True` silently returns None or falls back.  The script
  forces `attn_implementation="eager"` for the attention pass only, and asserts
  it got real weights rather than trusting the flag.

* **Measure in the model's native dtype.**  rho is a fine-grained statistic and
  4-bit weights move it.  bf16 or fp16 is fine; 4-bit is not a measurement of
  the model you are writing about.

* **Everything is pinned.**  Model revision, tokenizer revision, the text
  sample and its SHA-256, dtype, seed and sequence shape all go into the
  manifest.  A second edition re-runs this and diffs the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# ---------------------------------------------------------------- the sample
# A fixed, licence-clean sample.  Replace SAMPLE_PATH with your own corpus slice
# if you prefer; the manifest records the SHA-256 either way, so the measurement
# stays reproducible whatever you point it at.
SAMPLE_PATH = HERE / "sample.txt"


def load_sample() -> tuple[str, str]:
    text = SAMPLE_PATH.read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Manifest:
    model: str
    revision: str
    dtype: str
    device: str
    n_sequences: int
    seq_len: int
    seed: int
    sample_sha256: str
    transformers_version: str
    torch_version: str
    platform: str

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


# ------------------------------------------------------- Chapter 2, F-2.4
def measure_embeddings(model, tok, n_tokens: int = 40, seed: int = 0) -> dict:
    """Raw against mean-centred cosine on a sample of embedding rows.

    The claim under test (§2.4, M-2.1) is that a trained embedding cloud carries
    a common offset large enough to dominate any cosine you compute without
    removing it first.
    """
    import torch

    W = model.get_input_embeddings().weight.detach().to(torch.float32).cpu().numpy()
    rng = np.random.default_rng(seed)
    # sample from the frequent end of the vocabulary: rare rows are barely trained
    idx = np.sort(rng.choice(min(8000, W.shape[0]), size=n_tokens, replace=False))
    S = W[idx]

    def cos(M):
        Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
        return Mn @ Mn.T

    raw, cen = cos(S), cos(S - W.mean(0, keepdims=True))
    off = ~np.eye(n_tokens, dtype=bool)
    return {
        "raw": raw, "centred": cen, "token_ids": idx,
        "labels": np.array([tok.decode([int(i)]) for i in idx], dtype=object),
        "raw_offdiag_mean": float(raw[off].mean()),
        "centred_offdiag_mean": float(cen[off].mean()),
        "vocab_size": int(W.shape[0]), "d": int(W.shape[1]),
    }


# ------------------------------------------------------- Chapter 5, F-5.1
def measure_rho(model, tok, text: str, n_seq: int, seq_len: int, seed: int) -> dict:
    """|mean(x)| / RMS(x) at every normalisation site.

    Hooks the *input* to each norm module, which is the residual stream as the
    block sees it.  A decoder has two norm sites per layer plus a final one
    before the unembedding, so 2L + 1 in total (§5.1).
    """
    import torch

    sites: list[tuple[str, list[float]]] = []
    handles = []

    def hook(name):
        def fn(_mod, inputs, _out):
            x = inputs[0].detach().to(torch.float32)
            mean = x.mean(-1).abs()
            rms = x.pow(2).mean(-1).sqrt()
            sites_by_name[name].append((mean / (rms + 1e-12)).flatten().cpu().numpy())
        return fn

    norm_names = [n for n, m in model.named_modules()
                  if m.__class__.__name__.endswith(("RMSNorm", "LayerNorm"))
                  and "norm" in n.lower()]
    if not norm_names:
        raise SystemExit("no normalisation modules found; print model.named_modules() "
                         "and widen the filter for this architecture")
    sites_by_name = {n: [] for n in norm_names}
    for n in norm_names:
        handles.append(dict(model.named_modules())[n].register_forward_hook(hook(n)))

    batches = _batches(tok, text, n_seq, seq_len, seed)
    with torch.no_grad():
        for ids in batches:
            model(ids.to(model.device))
    for h in handles:
        h.remove()

    out = {}
    for n in norm_names:
        v = np.concatenate(sites_by_name[n])
        out[n] = {"median": float(np.median(v)), "p99": float(np.percentile(v, 99)),
                  "p1": float(np.percentile(v, 1)), "q1": float(np.percentile(v, 25)),
                  "q3": float(np.percentile(v, 75)), "n": int(v.size)}
    return {"sites": out, "n_sites": len(norm_names)}


# ------------------------------------------------------- Chapter 3, F-3.6
def measure_attention(model_id: str, revision: str | None, tok, text: str,
                      layer: int, seq_len: int, seed: int, dtype) -> dict:
    """Averaged attention weights for one layer, and the position-0 share.

    Reloads the model with eager attention, because a fused kernel never builds
    the matrix this figure is about.
    """
    import torch
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=dtype,
        attn_implementation="eager", low_cpu_mem_usage=True)
    m.eval()
    ids = _batches(tok, text, 1, seq_len, seed)[0].to(m.device)
    with torch.no_grad():
        out = m(ids, output_attentions=True)
    if out.attentions is None:
        raise SystemExit("attentions are None: the model ignored eager attention. "
                         "Check attn_implementation and the transformers version.")
    A = out.attentions[layer][0].to(torch.float32).cpu().numpy()   # (h, s, s)
    A_mean = A.mean(0)
    denom = np.arange(1, seq_len + 1)                # each row i sums to 1 over i+1 keys
    return {"map": A_mean, "layer": layer,
            "position0_share_per_head": A[:, :, 0].mean(1),
            "position0_share": float(A_mean[:, 0].mean()),
            "column_sums": A_mean.sum(0),
            "row_sum_check": float(np.abs(A_mean.sum(1) - 1).max()),
            "n_heads": int(A.shape[0]), "seq_len": int(seq_len),
            "_unused_denom": denom[:0]}


# ------------------------------------------------------------------ helpers
def _batches(tok, text: str, n_seq: int, seq_len: int, seed: int):
    import torch
    ids = tok(text, return_tensors="pt").input_ids[0]
    if ids.numel() < seq_len + 1:
        reps = int(np.ceil((seq_len + 1) / max(1, ids.numel())))
        ids = ids.repeat(reps)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, ids.numel() - seq_len, size=n_seq)
    return [ids[s:s + seq_len].unsqueeze(0) for s in starts]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face model id")
    ap.add_argument("--revision", default=None, help="commit SHA; pin it")
    ap.add_argument("--what", default="all",
                    choices=["all", "embeddings", "rho", "attention"])
    ap.add_argument("--n-seq", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--attn-layer", type=int, default=16)
    ap.add_argument("--attn-seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, a.dtype)
    device = (a.device if a.device != "auto" else
              "cuda" if torch.cuda.is_available() else
              "mps" if torch.backends.mps.is_available() else "cpu")
    DATA.mkdir(parents=True, exist_ok=True)
    text, sha = load_sample()

    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    print(f"loading {a.model} ({a.dtype}) on {device} ...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, revision=a.revision, dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.eval()

    if a.what in ("all", "embeddings"):
        r = measure_embeddings(model, tok)
        np.savez_compressed(DATA / "f02_04_cosine.npz", **{
            k: v for k, v in r.items() if isinstance(v, np.ndarray)})
        (DATA / "f02_04_summary.json").write_text(json.dumps(
            {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}, indent=2))
        print(f"F-2.4  raw off-diagonal mean {r['raw_offdiag_mean']:.3f}"
              f" -> centred {r['centred_offdiag_mean']:.3f}")

    if a.what in ("all", "rho"):
        r = measure_rho(model, tok, text, a.n_seq, a.seq_len, a.seed)
        (DATA / "f05_01_rho.json").write_text(json.dumps(r, indent=2))
        meds = [v["median"] for v in r["sites"].values()]
        print(f"F-5.1  {r['n_sites']} norm sites;"
              f" median rho min {min(meds):.4f}, max {max(meds):.4f}")
        if max(meds) >= 0.05:
            print("       NOTE: a site exceeds 0.05. That layer may genuinely use the"
                  " all-ones direction; §5.1's claim is falsifiable and this is the test.")

    if a.what in ("all", "attention"):
        del model
        r = measure_attention(a.model, a.revision, tok, text,
                              a.attn_layer, a.attn_seq_len, a.seed, dtype)
        np.savez_compressed(DATA / "f03_06_attention.npz", **{
            k: v for k, v in r.items() if isinstance(v, np.ndarray)})
        (DATA / "f03_06_summary.json").write_text(json.dumps(
            {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}, indent=2))
        print(f"F-3.6  layer {r['layer']}: position-0 share"
              f" {100*r['position0_share']:.1f}%  (row-sum error"
              f" {r['row_sum_check']:.2e})")

    import transformers
    Manifest(model=a.model, revision=a.revision or "unpinned", dtype=a.dtype,
             device=device, n_sequences=a.n_seq, seq_len=a.seq_len, seed=a.seed,
             sample_sha256=sha, transformers_version=transformers.__version__,
             torch_version=torch.__version__, platform=platform.platform()
             ).write(DATA / "manifest.json")
    print(f"\nwrote {DATA}/  — commit this directory, not the checkpoint")
    if a.revision is None:
        print("WARNING: --revision not given, so the manifest records 'unpinned'."
              " Pin it before the measurement goes into the book.")


if __name__ == "__main__":
    main()
