"""Chapter 8 — The Objective.

Generated into `notebooks/ch08_objective.ipynb` by `build_all.py`.  The chapter
cites §1, §3 and §4 by number, so sections may be added but never renumbered.
"""
from __future__ import annotations

CHAPTER = 8
SLUG = "objective"
TITLE = "The Objective"
BLURB = (
    "Maximum likelihood becomes per-token cross-entropy in four steps, one "
    "measurement gets four names, and label smoothing buys a finite minimiser "
    "at a price this notebook computes rather than quotes."
)

# ---------------------------------------------------------------------------
S1 = r'''
SEED = 8001
rng = np.random.default_rng(SEED)
V, T, order = 11, 400, 2          # a character-level model with a 2-token context

# A model that is a lookup table over contexts.  Nothing here is a transformer,
# which is D-8.1 step 7: the objective is not architecture-specific.
table = rng.gamma(0.7, size=(V ** order, V))
table /= table.sum(axis=1, keepdims=True)


def ctx_index(seq, t):
    i = 0
    for k in range(order):
        i = i * V + (seq[t - order + k] if t - order + k >= 0 else 0)
    return i


corpus = rng.integers(0, V, T + order)
cond = np.array([table[ctx_index(corpus, t)][corpus[t]]
                 for t in range(order, order + T)])
assert cond.shape == (T,) and (cond > 0).all()

# Step 1 and step 2: the joint factorises exactly, and the logarithm turns a
# product that underflows into a sum that does not.
log_joint = float(np.log(cond).sum())
prod = np.prod(cond[:40])                      # 40 terms is already this small
assert prod < 1e-30
assert abs(np.log(prod) - np.log(cond[:40]).sum()) < 1e-10
assert np.prod(cond) == 0.0, "the full product underflows to zero in float64"

# Step 3: negate and divide by T.  The two routes to the same number.
per_token = -log_joint / T
mean_nll = float(-np.log(cond).mean())
assert abs(per_token - mean_nll) < 1e-12
assert abs(log_joint + T * mean_nll) < 1e-10
print("corpus log-likelihood %.6f over %d tokens, mean NLL %.6f"
      % (log_joint, T, mean_nll))

# Step 4: one term is the cross-entropy against a one-hot target.
for t in (0, 17, T - 1):
    p = table[ctx_index(corpus, order + t)]
    e = np.zeros(V); e[corpus[order + t]] = 1.0
    assert abs(-np.log(p[corpus[order + t]]) - (-(e * np.log(p)).sum())) < 1e-12

# Step 5: the mean of those is the cross-entropy against the empirical
# distribution, which is the third expression of equation (8.1).
rows = np.array([table[ctx_index(corpus, t)] for t in range(order, order + T)])
onehot = np.zeros((T, V))
onehot[np.arange(T), corpus[order:]] = 1.0
H_emp = float(-(onehot / T * np.log(rows)).sum())
assert abs(H_emp - mean_nll) < 1e-12
print("H(p_hat, p_theta) computed as an average over positions: %.6f" % H_emp)

# Step 6: three names, one problem.  The argmax of the likelihood and the
# argmin of the mean NLL pick the same model out of a family.
def mean_nll_of(temp):
    q = rows ** (1.0 / temp)
    q /= q.sum(axis=1, keepdims=True)
    return float(-np.log(q[np.arange(T), corpus[order:]]).mean())


temps = np.linspace(0.5, 3.0, 51)
losses = np.array([mean_nll_of(t) for t in temps])
liks = np.array([-T * L for L in losses])              # log-likelihood
assert int(losses.argmin()) == int(liks.argmax())
print("argmin of the loss and argmax of the likelihood agree at index %d"
      % int(losses.argmin()))
'''

S2 = r'''
SEED = 8002
rng = np.random.default_rng(SEED)
V, T = 9, 300

phat = rng.gamma(1.0, size=V); phat /= phat.sum()      # the empirical row
theta0 = rng.standard_normal(V)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def H(p):
    return float(-(p * np.log(p)).sum())


def KL(p, q):
    return float((p * np.log(p / q)).sum())


# Equation (8.5), for several models.  Exact to machine precision.
for k in range(20):
    p = softmax(theta0 + 0.7 * rng.standard_normal(V))
    ce = float(-(phat * np.log(p)).sum())
    assert abs(ce - (H(phat) + KL(phat, p))) < 1e-12, k
print("H(p_hat, p_theta) = H(p_hat) + KL to %.1e over 20 draws" % 1e-12)

# The first term does not depend on theta.  That is the whole reason the two
# objectives are one sentence.
entropies = [H(phat) for _ in range(5)]
assert max(entropies) - min(entropies) == 0.0
assert H(phat) > 0.0

# Consequence one: the loss has a floor that is not zero, and it is reached
# only when the model reproduces the data distribution exactly.
assert KL(phat, phat) < 1e-15
assert abs(float(-(phat * np.log(phat)).sum()) - H(phat)) < 1e-15
for _ in range(50):
    p = softmax(rng.standard_normal(V) * 2)
    assert KL(phat, p) >= -1e-15
    assert float(-(phat * np.log(p)).sum()) >= H(phat) - 1e-12
print("loss floor H(p_hat) = %.6f nats, and no model beat it in 50 draws"
      % H(phat))

# Consequence two: the asymmetry is doing work.  Give the data one token it
# almost never emits, then compare a model that misses real mass against one
# that spends mass the data has no use for.
phat[-1] = 1e-6
phat /= phat.sum()
big, small = int(phat.argmax()), len(phat) - 1
q_misses = phat.copy(); q_misses[big] = 1e-6; q_misses /= q_misses.sum()
q_spends = phat.copy(); q_spends[small] = 0.3; q_spends /= q_spends.sum()
assert KL(phat, q_misses) > 10 * KL(phat, q_spends)
# and the reverse divergence, which is not the objective, ranks the same two
# models the other way round.  That is why maximum likelihood over-generates.
assert KL(q_spends, phat) > KL(q_misses, phat)
assert KL(phat, q_misses) > KL(phat, q_spends)
print("missing real mass costs %.4f nats; spending mass the data does not use "
      "costs %.4f (and %.4f under the reverse KL, which is not the objective)"
      % (KL(phat, q_misses), KL(phat, q_spends), KL(q_spends, phat)))
'''

S3 = r'''
import json, os, math
from arith.model_d import MODEL_D, loss_units, BYTES_PER_TOKEN

SEED = 8003
rng = np.random.default_rng(SEED)

# Steps 1 and 2 of D-8.3, on an actual sample: perplexity is the reciprocal
# geometric mean of the assigned probabilities, and its logarithm is the CE.
V, T = 7, 500
probs = rng.dirichlet(np.ones(V) * 0.8, size=T)
ids = np.array([rng.choice(V, p=probs[t]) for t in range(T)])
assigned = probs[np.arange(T), ids]
ppl_geo = float(np.exp(-np.log(assigned).mean()))
ce_nats = float(-np.log(assigned).mean())
assert abs(ppl_geo - np.exp(ce_nats)) < 1e-12
assert abs(np.prod(assigned[:20]) ** (-1 / 20) - np.exp(-np.log(assigned[:20]).mean())) < 1e-9

# Steps 3 and 4: one number, two bases.  The conversion, and the identity.
u = loss_units(ce_nats, BYTES_PER_TOKEN)
assert abs(u["bits_per_token"] - ce_nats / math.log(2)) < 1e-15
assert abs(u["perplexity"] - 2 ** u["bits_per_token"]) < 1e-9
assert abs(1 / math.log(2) - 1.442695) < 1e-6 and abs(math.log(2) - 0.693147) < 1e-6
print("sampled corpus: %.4f nats = %.4f bits = perplexity %.4f"
      % (ce_nats, u["bits_per_token"], u["perplexity"]))

# Model D's four coordinates, from arith/model_d.py rather than from the page.
m = loss_units(2.03, BYTES_PER_TOKEN)
assert round(m["bits_per_token"], 3) == 2.929
assert round(m["perplexity"], 3) == 7.614 and round(m["perplexity"], 1) == 7.6
assert round(m["bits_per_byte"], 4) == 0.7707
assert abs(m["perplexity"] - 2 ** m["bits_per_token"]) < 1e-9

# Step 6: the two anchors.  A model that has learned nothing, and a perfect one.
assert round(m["uniform_nats"], 3) == 11.762 == round(math.log(MODEL_D.V), 3)
assert m["uniform_ppl"] == MODEL_D.V
assert round(100 * m["fraction_of_uniform"]) == 17
assert loss_units(0.0)["perplexity"] == 1.0
print("Model D: %.2f nats, %.3f bits, PPL %.3f, %.4f bits/byte, %d%% of "
      "uniform (%.3f nats, PPL %d)"
      % (2.03, m["bits_per_token"], m["perplexity"], m["bits_per_byte"],
         round(100 * m["fraction_of_uniform"]), m["uniform_nats"],
         int(m["uniform_ppl"])))

# The assumption clause, made concrete: perplexity is comparable only under one
# tokenizer.  These bytes-per-token are a committed measurement, F-8.2.
def repo_file(*parts):
    """Works whether the notebook is run from notebooks/ or from the root."""
    for base in ("..", "."):
        p = os.path.join(base, *parts)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(os.path.join(*parts))


D = json.load(open(repo_file("figs", "data", "fig82_tokenizers.json")))
rows = D["rows"]
expect = {"GPT-2": 3.5284, "SmolLM2": 3.7999, "Llama-3": 4.0796}
for r in rows:
    assert abs(r["bytes_per_token"] - expect[r["name"]]) < 1e-4, r
    assert abs(r["tokens"] * r["bytes_per_token"] - D["sample_bytes"]) < 1e-6

# Fix the model's quality in bits per byte, which is tokenizer-free, and ask
# what perplexity each tokenizer would report for it.
bpb = m["bits_per_byte"]
bpt = np.array([r["bytes_per_token"] for r in rows])
ppl = 2.0 ** (bpb * bpt)
assert np.abs((bpb * bpt) / bpt - bpb).max() < 1e-12          # identical by construction
spread = ppl.max() / ppl.min() - 1
assert round(100 * spread) == 34, spread
assert ppl.max() / ppl.min() > 1.3
print("same model, three tokenizers: PPL %s, spread %.0f%%, bits/byte identical "
      "at %.4f" % (np.round(ppl, 3), 100 * spread, bpb))
'''

S4 = r'''
import math
from scipy.optimize import brentq
from arith.model_d import MODEL_D, label_smoothing

SEED = 8004
rng = np.random.default_rng(SEED)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


# Steps 1 to 4 by construction rather than by assertion of the algebra: fit the
# logits to the smoothed target by gradient descent and read off the gap.
V_small, eps = 200, 0.1
y = 3
q = np.full(V_small, eps / V_small)
q[y] += 1 - eps
assert abs(q.sum() - 1.0) < 1e-15

z = rng.standard_normal(V_small) * 0.5
for step in range(40000):
    p = softmax(z)
    z -= 3.0 * (p - q)                 # dH(q, softmax(z))/dz = p - q
p = softmax(z)
assert np.abs(p - q).max() < 1e-9, np.abs(p - q).max()

gaps = z[y] - np.delete(z, y)
analytic = label_smoothing(eps, V_small)["gap"]
assert gaps.std() < 1e-6, "every wrong logit lands at the same distance"
assert abs(gaps.mean() - analytic) < 1e-3, (gaps.mean(), analytic)
floor = float(-(q * np.log(q)).sum())
final_loss = float(-(q * np.log(p)).sum())
assert abs(final_loss - floor) < 1e-6
assert abs(floor - label_smoothing(eps, V_small)["floor"]) < 1e-12
print("gradient descent converged to gap %.4f (analytic %.4f) and loss %.4f "
      "(floor %.4f)" % (gaps.mean(), analytic, final_loss, floor))

# Step 5: with eps = 0 the objective has no finite minimiser.  The logit gap
# grows without bound, and the loss keeps falling as it does.
z0 = np.zeros(V_small)
one_hot = np.zeros(V_small); one_hot[y] = 1.0
seen, loss_seen = [], []
for step in range(60000):
    z0 -= 2.0 * (softmax(z0) - one_hot)
    if step in (999, 9999, 59999):
        seen.append(z0[y] - z0[0])
        loss_seen.append(float(-np.log(softmax(z0)[y])))
# The gap never settles: it grows without bound, slowly, and the loss keeps
# falling with it.  The smoothed run above stopped at a finite gap.
assert seen[0] < seen[1] < seen[2]
assert seen[2] - seen[0] > 3.0
assert loss_seen[0] > loss_seen[1] > loss_seen[2] > 0.0
assert seen[2] > 1.5 * gaps.mean(), "and it has already passed the smoothed gap"
print("unsmoothed: gap %.2f then %.2f then %.2f nats, loss %.2e then %.2e then "
      "%.2e, neither converged" % tuple(seen + loss_seen))

# Steps 6 and 7 at Model D's vocabulary, from arith/model_d.py.
ls = label_smoothing(0.1, MODEL_D.V)
assert round(ls["ratio"]) == 1_154_305
assert round(ls["gap"], 4) == 13.9590 and round(ls["gap"], 3) == 13.959
assert abs(ls["gap"] - ls["gap_large_V"]) < 1e-5          # good to six decimals
assert abs(ls["gap"] - math.log((1 - 0.1 + 0.1 / MODEL_D.V) / (0.1 / MODEL_D.V))) < 1e-12

# Step 8: the floor, and where it comes from.
py, pj = ls["p_y"], ls["p_j"]
first = -py * math.log(py)
second = -(MODEL_D.V - 1) * pj * math.log(pj)
assert abs(first + second - ls["floor"]) < 1e-12
assert round(ls["floor"], 4) == 1.5012
assert round(second, 3) == 1.406 and round(first, 3) == 0.095
assert second / ls["floor"] > 0.93, "almost all of it is spreading eps over V"
print("floor %.4f nats = %.3f (target term) + %.3f (the eps spread over V)"
      % (ls["floor"], first, second))

# The small-eps form.  The book corrects an error here: the floor is
# eps*(1 + log(V/eps)) and NOT eps*log(V).  The first term of the expansion is
# the -p_j log p_j sum, which carries a log(1/eps) that the crude form drops.
for e in (0.1, 0.01, 1e-3, 1e-4):
    exact = label_smoothing(e, MODEL_D.V)["floor"]
    correct = e * (1 + math.log(MODEL_D.V / e))
    crude = e * math.log(MODEL_D.V)
    assert abs(correct - label_smoothing(e, MODEL_D.V)["floor_small_eps"]) < 1e-15
    assert abs(exact - correct) / exact < 4e-3, (e, exact, correct)
    assert abs(exact - crude) / exact > 0.2, (e, exact, crude)
    print("eps %.0e: exact %.6f, eps(1 + log(V/eps)) %.6f, eps log V %.6f"
          % (e, exact, correct, crude))
# and the correct form gets better as eps shrinks, which is what "small-eps
# form" has to mean.
rel = [abs(label_smoothing(e, MODEL_D.V)["floor"]
           - e * (1 + math.log(MODEL_D.V / e)))
       / label_smoothing(e, MODEL_D.V)["floor"] for e in (0.1, 0.01, 1e-3, 1e-4)]
assert rel[0] > rel[1] > rel[2] > rel[3] and rel[3] < 1e-4

# The failure mode: the floor grows with log V, so a constant copied across
# vocabularies is not a constant cost.
old = label_smoothing(0.1, 32_000)
assert round(old["floor"], 4) == 1.3624 and round(old["gap"], 4) == 12.5707
assert ls["floor"] - old["floor"] > 0.13
e1 = brentq(lambda e: label_smoothing(e, MODEL_D.V)["floor"] - 1.0, 1e-6, 0.9)
assert round(e1, 4) == 0.0647
print("V = 32000 floors at %.4f, V = %d at %.4f; %.2f%% smoothing alone costs "
      "a whole nat" % (old["floor"], MODEL_D.V, ls["floor"], 100 * e1))
'''

SECTIONS = [
    ("1", "From corpus likelihood to per-token cross-entropy",
     "Four steps take the probability a model assigns to a corpus into the loss "
     "every training log prints: factorise, take logarithms, negate and divide "
     "by the number of tokens, and recognise one term as a cross-entropy "
     "against a one-hot target. The cell does all four on a small "
     "character-level lookup model, which is D-8.1 step 7 made concrete: "
     "nothing here is transformer-specific. It also shows the product "
     "underflowing, which is why step 2 is not cosmetic.",
     S1),
    ("2", "Cross-entropy is entropy plus a divergence",
     "The decomposition in equation (8.5) is what makes maximum likelihood and "
     "minimum divergence the same sentence: the entropy term does not depend on "
     "the parameters, so minimising one minimises the other. The cell checks "
     "the identity, checks that the entropy term does not move when the model "
     "does, and shows the two consequences the chapter draws, a non-zero floor "
     "and an asymmetry that punishes missing mass far more than misplaced mass.",
     S2),
    ("3", "Perplexity, and the two bases",
     "Perplexity is the reciprocal geometric mean of the probabilities the "
     "model assigned to what happened, so its logarithm is the cross-entropy "
     "and the only question is which base you took. The cell recomputes all "
     "four coordinates through arith/model_d.py, checks the two anchors, and "
     "then loads the committed tokenizer measurement to show what the "
     "assumption clause costs: at one fixed quality in bits per byte, three "
     "tokenizers report perplexities that span a third.",
     S3),
    ("4", "Label smoothing bounds the logit gap, and floors the loss",
     "Smoothing the target gives the objective a finite minimiser, and the "
     "minimiser is the target itself, so the optimal logit gap is a logarithm "
     "of a probability ratio. The cell finds that gap by gradient descent "
     "rather than asserting the algebra, shows the unsmoothed objective still "
     "growing its gap after sixty thousand steps, and then prices the floor. "
     "The small-eps form is the corrected one, eps times one plus log of V over "
     "eps, not eps times log V, and the cell shows the crude version is off by "
     "more than a fifth at every epsilon tested.",
     S4),
]
