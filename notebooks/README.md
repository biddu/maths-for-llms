# Notebooks

One notebook per chapter, named `chNN_<slug>.ipynb`, each reproducing every
numbered equation in that chapter with a numerical check. The check is the
point: a cell that prints a plot and asserts nothing is not doing its job.

| Notebook | Chapter | Status |
|---|---|---|
| `ch01_toolkit.ipynb` | 1 — The Toolkit in One Chapter | to write |
| `ch02_embeddings.ipynb` | 2 — Embeddings as Geometry | to write |
| `ch03_attention.ipynb` | 3 — Attention from First Principles | to write |
| `ch04_position.ipynb` | 4 — Position | to write |
| `ch05_norm_residual.ipynb` | 5 — Normalisation and the Residual Stream | to write |
| `ch06_ffn.ipynb` | 6 — The Feed-Forward Block | to write |
| `ch07_backprop.ipynb` | 7 — Backpropagation Through a Transformer Block | to write |
| `ch08_objective.ipynb` | 8 — The Objective | to write |
| `ch09_optimisation.ipynb` | 9 — Optimisation | to write |

Every notebook is executed top to bottom by CI on every commit. NumPy and
PyTorch only, at the versions pinned in `requirements.txt`.


## `ch06_ffn.ipynb` — the cell list

1. **Key--value.** One Model D FFN layer. Verify
   `FFN(x) == sum_j phi(x @ W_up[:,j]) * W_down[j,:]` to 1e-5 by explicit
   summation, and report how many units clear threshold on a real token.
   Sparsity is a measurement here, not an assertion.
2. **GeLU.** Assert the minimum is -0.1700 +- 1e-4 at -0.7518 +- 1e-4, and that
   `max |exact - tanh|` on [-8, 8] is below 1e-3. Print where the maximum is.
3. **Bilinearity.** Assert `glu(x)_j == 0.5 * x @ outer(w_j, v_j) @ x.T` to 1e-6
   in the small-x limit; assert the ungated second-order matrix is symmetric to
   1e-9 and the gated one is not; reproduce the rank-2 eigen-pairing of
   equation (6.13).
4. **Parameter identity.** Assert `|3*d*(2*d_ff//3) - 2*d*d_ff| <= 3*d`,
   reproduce the box's five steps, and assert `intermediate_size == 14336`.
5. **Where the parameters are.** Recompute the non-embedding total from
   scratch, assert 6.9796e9 to 0.1% and an FFN share of 0.808 +- 0.001. This is
   the cell Chapter 12 imports.


## `ch07_backprop.ipynb` — the cell list

Float64 NumPy throughout, at reduced Model-D shapes (d = 128, h = 4, n_kv = 2,
d_ff = 352, s = 64), with a PyTorch reference alongside.

1. **Setup.** One pre-norm block forward; assert the NumPy and PyTorch forwards
   agree to 1e-12.
2. **Linear (D-7.1).** dW and dA against autograd to 1e-12. Then the tied
   embedding counterexample: assert the one-term version differs from the
   two-term version by more than 0.1 relative Frobenius.
3. **Softmax (D-7.2).** Equation (7.6) against the explicit Jacobian to 1e-12;
   the arithmetic and working-set table behind F-7.2.
4. **RMSNorm.** Assert the Jacobian is symmetric, that the projector is
   idempotent to 1e-14, and that it annihilates x to 1e-14.
5. **SwiGLU (7.10)–(7.14).** Against autograd to 1e-12. Separately assert the
   two paths into `n2bar` are not the same tensor, which guards a copy-paste
   bug the shapes cannot catch.
6. **Attention (D-7.3).** Full (dQ, dK, dV, dW) against autograd to 1e-12, MHA
   and GQA. Assert dS is exactly zero at every masked entry, and that the group
   contributions sum rather than average.
7. **Residual flow (D-7.4).** A 32-block stack, gradient norm by depth, pre-
   and post-norm, with and without the 1/sqrt(2L) residual scaling. This cell
   writes `figs/data/fig73.npz`.
8. **Mixed precision.** 10^4 Adam steps at eta = 3e-6 on a bf16 weight: assert
   it never moves, and that an fp32 master moves by 0.0298. Then the second
   moment: assert a bf16 `v` fails to track a tenfold drop in gradient scale
   while an fp32 `v` tracks it, leaving Adam's denominator 10x too large.
   Then fp16 gradient underflow and the loss-scaling fix.
9. **Arithmetic.** From `arith/model_d.py`: assert the block total is exactly
   5,435,883,520 bytes, that P is 79.0% +- 0.1%, and that the checkpoint
   optimum is clipped at s = 8192 and interior once attention is fused. Emits
   F-7.4 and F-7.5.


## `ch08_objective.ipynb` — the cell list

1. **Likelihood to cross-entropy.** A small character-level model; compute the
   corpus log-likelihood two ways, as a product of conditionals and as mean
   per-token CE times T, and assert 1e-10.
2. **CE = H(p-hat) + KL.** Assert the decomposition to 1e-12 and that H(p-hat)
   does not move when theta does.
3. **Units.** Round-trip 2.03 nats through all four coordinates; assert
   exp(2.03) = 2^(2.03/ln2) = 7.6141 to 1e-12 and that the table matches
   `arith/model_d.py::loss_units` exactly. Emits F-8.1's numbers.
4. **Tokenizer invariance.** Tokenize the committed held-out sample with GPT-2,
   SmolLM2 and Llama-3; assert the bytes-per-token are 3.5284, 3.7999 and
   4.0796, and that at fixed bits-per-byte the perplexities spread by 34%.
   Emits F-8.2 and `figs/data/fig82_tokenizers.json`.
5. **Label smoothing.** Fit logits by gradient descent on the smoothed target;
   assert the converged gap matches 13.959 to 1e-3 and the converged loss
   matches 1.5012 to 1e-4. Emits F-8.3.
6. **Causal equivalence.** Parallel masked forward against a sequential loop;
   assert the maximum per-token loss difference is below 1e-6 and report the
   wall-clock ratio (about 30x at s = 128 in NumPy).
7. **The cost of a tenth of a nat.** Assert `scaling_loss(N, D)["L"]` = 2.0323
   to 1e-4, so the box's 2.03 is a prediction and not a transcription. Then
   assert the multipliers 5.85, 6.64 and 3.03, the token figures 87.8T, 99.7T
   and 45.4T, and the headline: the data term is 0.0314 nats and the D to
   infinity floor at fixed N is 2.0009, so neither 0.1-nat target is reachable
   by tokens alone.


## `ch09_optimisation.ipynb` — the cell list

1. **Norm balls and oracles.** Brute-force the minimum of <g, D> over the l2,
   l-infinity and spectral balls by sampling; assert each is within 1e-3 of the
   closed forms of D-9.1 and D-9.4.
2. **EMA bias.** Simulate 10^4 runs of m_t on i.i.d. gradients with a known
   mean; assert |mean(m_t) / ((1 - b1^t) g_bar) - 1| < 0.02 for t = 1..50, and
   that the corrected estimator is unbiased to Monte-Carlo error.
3. **L2 is not decoupled.** Two optimisers, identical lambda; assert the
   realised per-coordinate decay spans more than an order of magnitude for
   L2-Adam and is constant to 1e-12 for AdamW.
4. **Newton-Schulz.** Assert the singular vectors are preserved to 1e-10 (they
   are exact, not approximate) and the singular values land in [0.68, 1.21]
   after five steps; time it against a reference SVD.
5. **Schedules.** Generate warmup-cosine and warmup-stable-decay arrays; assert
   the WSD trunk is bitwise identical across three branch budgets. That is the
   branch property, made a test.
6. **Clipping.** Assert global-norm clipping satisfies ||g|| <= c exactly and
   leaves the direction unchanged to 1e-12; assert per-coordinate clipping does
   not.
7. **The burst bound.** Reproduce equation (9.21) across a decade of beta2 and
   assert agreement with (1 - b1)/sqrt(1 - b2) to 1e-3. Emits F-9.5.
8. **Arithmetic.** Recompute the whole of §9.8 from `arith/model_d.py`; assert
   the AdamW total is 128.48 GB and Muon's is 100.57 GB to 0.01 GB.
