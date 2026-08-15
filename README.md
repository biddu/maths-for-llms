# Checkpoint measurements

Three figures depend on statistics of a trained model rather than on arithmetic:

| Figure | Chapter | The claim under test |
|---|---|---|
| F-2.4 | 2 | The embedding cloud carries an offset large enough to dominate any cosine computed without removing it |
| F-3.6 | 3 | Attention mass concentrates on position 0 for queries with no strong match |
| F-5.1 | 5 | `|mean(x)| / RMS(x)` of the residual stream is O(10⁻²) at every layer |

None of the three can go into print until the measurement exists. Chapter 5's
write plan says so explicitly, and it is right: the arithmetic box states its
claim in a falsifiable form, and a falsifiable claim with no measurement behind
it is not yet a claim.

## Which checkpoint

**Model D is defined by hyperparameters and never by name.** That was a
deliberate choice (CONVENTIONS §3) and it pays off here: the arithmetic boxes
need no checkpoint at all, because they are computed from the table. What the
three figures test is not Model D specifically but a distributional property of
trained decoder transformers. So measure on a model anyone can download without
accepting a licence, and name it in the caption rather than implying it is
Model D.

Configs read from the Hub, against Model D's row:

| Model | L | d | h | n_kv | d_ff | V | RoPE base | Licence |
|---|---|---|---|---|---|---|---|---|
| **Model D (the book)** | 32 | 4096 | 32 | 8 | 14336 | 128256 | 500,000 | — |
| Llama-3-8B | 32 | 4096 | 32 | 8 | 14336 | 128256 | 500,000 | **gated** |
| Qwen3-8B-Base | 36 | 4096 | 32 | 8 | 12288 | 151936 | 1,000,000 | Apache 2.0 |
| OLMo-2-7B | 32 | 4096 | 32 | **32** | 11008 | 100352 | 500,000 | Apache 2.0 |

No ungated model matches Model D exactly. Llama-3-8B does and is gated, which
would break the promise that a reader can check the book on a fresh clone.

**Recommendation: measure on Qwen3-8B-Base and OLMo-2-7B, and report both.**
Two models agreeing is much stronger evidence than one, it costs a second run,
and it protects the claim from being an artefact of one training recipe. Qwen3
matches Model D's grouped-query structure; OLMo-2 is fully open including its
training data, which is what to point at if a reviewer asks whether the
measurement is contaminated. Note in the caption that OLMo-2 is multi-head
rather than grouped-query, since Chapter 3's figure is about attention.

## Where to run it

**Not in the Cowork workspace.** That VM has 3 GB of memory, no network and no
PyTorch; it is for editing files, not for loading an 8 B model.

Run it from a normal Terminal on the Mac, or on a rented GPU:

* **Mac, 24 GB unified memory or more.** `--dtype float16 --device mps`. An 8 B
  model is 16 GB of weights in 16-bit, so 24 GB is comfortable and 16 GB is not.
* **Mac, 16 GB.** Use the 7 B at `--dtype float16` (14 GB) and expect swapping,
  or run on CPU and wait. The work is small: 256 sequences of 512 tokens is a
  few minutes on a GPU and well under an hour on CPU.
* **Rented GPU.** Any 24 GB card. The whole job is one hour of compute at most,
  which is a rounding error against the cover budget.

## Running it

    pip install torch transformers numpy
    python measure/checkpoint_stats.py \
        --model Qwen/Qwen3-8B-Base \
        --revision <commit-sha> \
        --dtype float16 --device mps \
        --n-seq 256 --seq-len 512 --attn-layer 18

`--revision` is not optional in spirit. Without it the manifest records
`unpinned` and the measurement is not reproducible; a model card can be updated
under you. Take the SHA from the Hub's commit history.

The script writes small derived arrays to `measure/data/` plus a
`manifest.json` recording the model, revision, dtype, device, sample SHA-256,
sequence shape, seed and library versions. **Commit that directory. Never commit
the checkpoint.** The figures then rebuild from the committed statistics on a
fresh clone and in CI without a download, and a second edition re-runs the
script and diffs the manifest.

## What to watch for

**Attention weights need eager attention.** SDPA and FlashAttention never
materialise the s × s matrix, which is exactly the point of them (§11.6), so
`output_attentions=True` returns `None` or silently falls back. The script
forces `attn_implementation="eager"` for the attention pass and asserts it got
real weights rather than trusting the flag. This is the single most common way
this measurement goes wrong.

**Measure in the model's native precision.** ρ is a fine-grained statistic and
4-bit weights move it. bf16 or fp16 is fine. A 4-bit measurement is a
measurement of the quantised model, which is a different object and is
Chapter 13's subject, not Chapter 5's.

**Count the norm sites.** A decoder has 2L + 1: two per layer plus the final
norm before the unembedding. If the hook finds a different number, the filter is
catching something else, and the script prints the count so you notice.

**Check the row sums.** Attention rows must sum to 1 by
\eqref{eq:simplexconstraint}. The script reports the maximum deviation; if it is
not at the level of floating-point error, the mask or the slicing is wrong.

## Validation run

The script was validated end to end on SmolLM2-135M, which is small enough to
run anywhere and real enough to test the code path:

    F-2.4  raw off-diagonal cosine 0.245 -> mean-centred 0.124
    F-5.1  61 norm sites (= 2 x 30 + 1), median rho from 0.0172 to 0.0417
    F-3.6  layer 8: position-0 share 13.2%, row-sum error 1.8e-07

Two things to take from that. Chapter 5's claim survives: ρ is O(10⁻²) at every
site, on a real model. And Chapter 2's claim needs strengthening, which is the
subject of the next section.

## A correction the validation run produced

Chapter 2's notebook spec asserts `raw mean > 0.2 and centred mean < 0.05`. On
SmolLM2-135M the raw off-diagonal cosine is 0.245, so the first half holds, and
the mean-centred value is 0.124, so **the second half fails**. Removing the mean
takes out about half the anisotropy and leaves the rest.

Removing the mean *and* the top principal direction takes it to 0.003:

| Treatment | Mean off-diagonal cosine |
|---|---|
| raw | +0.245 |
| mean-centred | +0.124 |
| mean + top-1 principal direction removed | **+0.003** |
| mean + top-10 principal directions removed | +0.003 |

The top singular direction alone carries 23% of the squared Frobenius norm of
the centred embedding matrix, and ‖μ‖ is 0.69 of a typical row norm. So the
cloud has two problems, not one: an offset and a dominant direction. This is
exactly Mu and Viswanath's *all-but-the-top* result, which Chapter 2 already
cites in its source list but under-states in the text.

The consequence for the book: §2.4's misconception box and F-2.4 should say that
centring is necessary and not sufficient, `centred_cosine` in the notebook
should remove the mean and the top direction, and the assertion should be
`< 0.05` after both rather than after centring alone. One extra table row in
F-2.4 makes the point better than the current two panels do.
