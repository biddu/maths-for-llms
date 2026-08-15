# Notebooks

One notebook per chapter, each reproducing what that chapter derives and
**asserting** it. The assertion is the point: a cell that prints a number and
checks nothing is not doing its job, and CI fails any section that does it.

| Notebook | Chapter | Sections | Asserts |
|---|---|---:|---:|
| `ch01_toolkit.ipynb` | 1 — The Toolkit in One Chapter | 6 | 85 |
| `ch02_embeddings.ipynb` | 2 — Embeddings as Geometry | 4 | 50 |
| `ch03_attention.ipynb` | 3 — Attention from First Principles | 4 | 61 |
| `ch04_position.ipynb` | 4 — Position | 5 | 84 |
| `ch05_norm_residual.ipynb` | 5 — Normalisation and the Residual Stream | 5 | 48 |
| `ch06_ffn.ipynb` | 6 — The Feed-Forward Block | 4 | 71 |
| `ch07_backprop.ipynb` | 7 — Backpropagation Through a Block | 6 | 57 |
| `ch08_objective.ipynb` | 8 — The Objective | 4 | 62 |
| `ch09_optimisation.ipynb` | 9 — Optimisation | 4 | 76 |
| `ch10_scaling.ipynb` | 10 — Scaling Laws | 3 | 71 |
| `ch11_cost_of_attention.ipynb` | 11 — The Cost of Attention | 5 | 88 |
| `ch12_moe.ipynb` | 12 — Mixture of Experts | 4 | 80 |
| `ch13_quant_lora.ipynb` | 13 — Quantisation and Low-Rank Adaptation | 5 | 73 |
| `ch14_decoding.ipynb` | 14 — Generation and Decoding | 3 | 76 |
| `ch15_post_training.ipynb` | 15 — Post-Training Mathematics | 5 | 77 |
| `ch16_representation.ipynb` | 16 — What the Model Represents | 4 | 74 |
| | | **71** | **1133** |

The section numbers are a **contract**, not a convention. Each chapter carries
`\repo` margin notes of the form `▸ ch03_attention.ipynb §2` beside the
derivation that section reproduces, so a section that moves breaks a printed
cross-reference. There are 56 such notes across the sixteen chapters.

## Running them

    pip install -r ../requirements.txt
    jupyter lab                      # or execute headless, as CI does

Nothing here downloads a checkpoint and nothing needs a GPU. The whole set
executes in about eighty seconds on a laptop.

## Three rules, and CI enforces all three

1. **Every section asserts.** CI executes every notebook top to bottom, then
   separately checks that each section cell contains an `assert`. A notebook
   that runs but proves nothing fails the second check.
2. **NumPy and SciPy only.** No `torch`, no `transformers`, no plotting, no
   network. Where a measurement is genuinely needed it is read from
   `../figs/data/`, which is committed: a trained layer's scores, the
   gradient-flow ratios, the MoE load histograms, the tokenizer byte counts,
   eight real logit positions, and a real draft-target acceptance measurement.
3. **Numbers come from `arith/`, never from the page.** A notebook that
   hard-codes `6_979_588_096` agrees with the book right up until someone edits
   the model table. One that calls `non_embedding(MODEL_D)` cannot.

## They are generated, and that is deliberate

Each notebook is built from `nb_chNN.py` in this directory:

    python3 notebooks/build_all.py           rebuild all sixteen
    python3 notebooks/build_all.py 3 11      rebuild those two

Edit the generator, not the `.ipynb`. This is the same contract the figures
keep, and for the same reason: a second edition should be a re-run rather than a
re-check, and a notebook edited by hand drifts silently from the chapter it
claims to reproduce.

## What they are for

Not tutorials. The chapters teach; these check. Read one when you want to know
whether a result really holds, or when you want to change a constant and watch
what moves. The most useful ones for that are `ch10_scaling.ipynb`, where you
can substitute your own fitted exponents and see the compute-optimal frontier
move, and `ch11_cost_of_attention.ipynb`, where the cache arithmetic responds to
your own model's shape.
