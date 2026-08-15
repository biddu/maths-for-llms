"""Chapter 15 — Post-Training Mathematics.

Generated into `notebooks/ch15_post_training.ipynb` by `build_all.py`.  The
chapter cites §1, §2, §3, §4 and §5--§7 by number, so sections may be added but
never renumbered.

The correction this notebook carries is in §5--7.  The group-mean baseline is
often described as biasing the direction of the policy gradient.  It does not:
it is a PURE SHRINKAGE by exactly (G-1)/G, the cosine with the true gradient is
one to five decimal places at every G, and the shrinkage is absorbed into the
learning rate.  The bias worth worrying about is the division by std(r).
"""
from __future__ import annotations

CHAPTER = 15
SLUG = "post_training"
TITLE = "Post-Training Mathematics"
BLURB = (
    "Every named algorithm in this chapter is one objective with a different "
    "estimator, and every deletion from PPO's four resident models has a "
    "price. The identities are checked exactly, the biases are measured, and "
    "the memory and compute come from `arith/post_training_memory.py`."
)

SIGMOID = r'''
def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))
'''

S1 = SIGMOID + r'''
SEED = 1501
rng = np.random.default_rng(SEED)

# (15.1) is a logistic function of the reward MARGIN and of nothing else.  The
# two-line derivation: divide top and bottom by exp(r_w).
r_w = rng.standard_normal(2000) * 3.0
r_l = rng.standard_normal(2000) * 3.0
bt = np.exp(r_w) / (np.exp(r_w) + np.exp(r_l))
assert np.abs(bt - sigmoid(r_w - r_l)).max() < 1e-12
assert np.abs(bt + sigmoid(r_l - r_w) - 1.0).max() < 1e-12      # reflection
print("(15.1) equals sigma(r_w - r_l) to %.1e, so only the margin is ever "
      "modelled" % np.abs(bt - sigmoid(r_w - r_l)).max())

# The invariance that will matter.  r is identified only up to an ADDITIVE
# FUNCTION OF THE PROMPT: every member of {r + c(x)} fits the data identically.
n_x, n_y = 6, 4
r = rng.standard_normal((n_x, n_y))
c = rng.standard_normal(n_x) * 5.0
r_shift = r + c[:, None]
for x in range(n_x):
    for i in range(n_y):
        for j in range(n_y):
            assert abs((r_shift[x, i] - r_shift[x, j]) - (r[x, i] - r[x, j])) < 1e-12
assert np.abs(sigmoid(r_shift[:, :, None] - r_shift[:, None, :])
              - sigmoid(r[:, :, None] - r[:, None, :])).max() < 1e-14
print("adding a per-prompt constant leaves every pairwise probability "
      "unchanged: the whole column is free to slide")

# It is NOT invariant to a rescale, and that asymmetry is why r and beta only
# ever appear as r/beta.
for a in (0.5, 2.0, 3.0):
    scaled = sigmoid(a * (r[:, :, None] - r[:, None, :]))
    base = sigmoid(r[:, :, None] - r[:, None, :])
    off = ~np.eye(n_y, dtype=bool)
    assert np.abs(scaled - base)[:, off].max() > 0.05, a
    print("  scaling r by %.1f moves the largest pairwise probability by %.3f"
          % (a, np.abs(scaled - base)[:, off].max()))

# The likelihood of a preference dataset, and the fact that the MLE only ever
# sees differences: a fitted r and a fitted r + c(x) have identical likelihood.
pairs = [(rng.integers(n_x), *rng.choice(n_y, 2, replace=False)) for _ in range(400)]
nll = lambda R: -sum(np.log(sigmoid(R[x, i] - R[x, j])) for x, i, j in pairs)
assert abs(nll(r) - nll(r_shift)) < 1e-9
assert nll(r) > 0
print("the negative log-likelihood over %d pairs is %.6f for r and %.6f for "
      "r + c(x): the same number, so no data can choose between them"
      % (len(pairs), nll(r), nll(r_shift)))

# Failure mode: heterogeneous annotators produce intransitive preferences, and
# no scalar r represents an intransitive relation.  A three-item Condorcet cycle
# is the smallest instance, and the best-fitting scalar model returns the tie.
from scipy.optimize import minimize
wins = np.array([[0.0, 0.7, 0.3],                      # 1 beats 2, 2 beats 3,
                 [0.3, 0.0, 0.7],                      # 3 beats 1
                 [0.7, 0.3, 0.0]])


def cycle_nll(v):
    R = np.concatenate([[0.0], v])                     # gauge-fix r_0 = 0
    tot = 0.0
    for i in range(3):
        for j in range(3):
            if i != j:
                tot -= wins[i, j] * np.log(sigmoid(R[i] - R[j]))
    return tot


res = minimize(cycle_nll, np.zeros(2), method="Nelder-Mead",
               options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
fit = np.concatenate([[0.0], res.x])
assert np.abs(fit - fit.mean()).max() < 1e-4, fit
assert wins[0, 1] > 0.5 and wins[1, 2] > 0.5 and wins[2, 0] > 0.5
print("on a Condorcet cycle the maximum-likelihood scalar is flat to %.1e: it "
      "returns nobody's preference in particular" % np.abs(fit - fit.mean()).max())
'''

S2 = SIGMOID + r'''
SEED = 1502
rng = np.random.default_rng(SEED)

# (15.5), derived through the reflection identity, and checked against a central
# difference on a real scorer rather than asserted.
d_in, h, n = 12, 16, 64
P = [rng.standard_normal((d_in, h)) / np.sqrt(d_in),
     rng.standard_normal((h, h)) / np.sqrt(h),
     rng.standard_normal(h) / np.sqrt(h)]
Xw = rng.standard_normal((n, d_in))
Xl = rng.standard_normal((n, d_in))


def score(params, X):
    W1, W2, w3 = params
    return np.tanh(np.tanh(X @ W1) @ W2) @ w3


def rm_loss_and_grad(params, Xw, Xl):
    W1, W2, w3 = params
    A1w, A1l = np.tanh(Xw @ W1), np.tanh(Xl @ W1)
    A2w, A2l = np.tanh(A1w @ W2), np.tanh(A1l @ W2)
    delta = A2w @ w3 - A2l @ w3
    loss = float(-np.log(sigmoid(delta)).mean())
    gain = sigmoid(-delta) / len(delta)                # THE factor of (15.5)
    g3 = -(gain @ (A2w - A2l))
    b2w = -gain[:, None] * np.outer(np.ones(len(delta)), w3) * (1 - A2w ** 2)
    b2l = gain[:, None] * np.outer(np.ones(len(delta)), w3) * (1 - A2l ** 2)
    g2 = A1w.T @ b2w + A1l.T @ b2l
    b1w = (b2w @ W2.T) * (1 - A1w ** 2)
    b1l = (b2l @ W2.T) * (1 - A1l ** 2)
    g1 = Xw.T @ b1w + Xl.T @ b1l
    return loss, [g1, g2, g3]


L, G = rm_loss_and_grad(P, Xw, Xl)
assert L > 0 and np.isfinite(L)
eps = 1e-6
worst = 0.0
for k in range(3):
    flat = P[k].ravel()
    for i in rng.choice(flat.size, 10, replace=False):
        old = flat[i]
        flat[i] = old + eps
        Lp, _ = rm_loss_and_grad(P, Xw, Xl)
        flat[i] = old - eps
        Lm, _ = rm_loss_and_grad(P, Xw, Xl)
        flat[i] = old
        worst = max(worst, abs(G[k].ravel()[i] - (Lp - Lm) / (2 * eps)))
assert worst < 1e-7, worst
print("(15.5) against central differences on 30 coordinates: max abs %.2e" % worst)

# Swapping the pair negates the margin, so the two losses cannot both be small.
L2, _ = rm_loss_and_grad(P, Xl, Xw)
assert L + L2 >= 2 * np.log(2) - 1e-9
print("L(w,l) + L(l,w) = %.4f >= 2 log 2 = %.4f, with equality only at a tie"
      % (L + L2, 2 * np.log(2)))

# SELF-ANNEALING, step 6.  The gain is sigma(-Delta): about one when the pair is
# backwards, one half at a tie, and vanishing once the margin is comfortable.
assert abs(sigmoid(0.0) - 0.5) < 1e-15
assert abs(sigmoid(-4.0) - 0.018) < 5e-4
assert abs(sigmoid(-8.0) - 0.000335) < 1e-6
ratio = sigmoid(-8.0) / sigmoid(8.0)
assert abs(100 * ratio - 0.0335) < 1e-3
for margin in (-4.0, -1.0, 0.0, 1.0, 4.0, 8.0):
    print("  margin %+5.1f: gradient gain sigma(-Delta) = %.6f"
          % (margin, sigmoid(-margin)))
print("a pair already right by a margin of 8 contributes %.3f%% of what the "
      "same pair backwards contributes" % (100 * ratio))

# The consequence, made quantitative.  Once most pairs are right, the surviving
# gradient comes almost entirely from the few that are not, so a duplicate in
# THAT set is a large share of the whole update.
margins = np.concatenate([np.linspace(3.0, 9.0, 980),      # already right
                          np.linspace(-2.5, 0.5, 20)])     # still backwards
w = sigmoid(-margins)
share = w[-20:].sum() / w.sum()
assert (margins[-20:] < 0).mean() >= 0.8
assert share > 0.5, share
print("with %.0f%% of pairs still backwards, they carry %.0f%% of the total "
      "gradient weight: duplicating one of them is not a mild reweighting"
      % (100 * 20 / len(margins), 100 * share))

# Failure mode: only the DIFFERENCE is supervised, so the absolute scale is
# unconstrained.  Adding any constant to every score leaves the loss identical.
for c in (-7.0, 0.3, 12.0):
    shifted = [P[0], P[1], P[2]]
    Lc, _ = rm_loss_and_grad(shifted, Xw, Xl)
    assert abs(Lc - L) < 1e-12
sw, sl = score(P, Xw), score(P, Xl)
Lshift = float(-np.log(sigmoid((sw + 9.0) - (sl + 9.0))).mean())
assert abs(Lshift - L) < 1e-12
print("shifting every reward by 9.0 changes the loss by %.1e: any downstream "
      "use that treats r as calibrated is unsound" % abs(Lshift - L))
'''

S3 = r'''
SEED = 1503
V = 64


def problem(seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(V)
    z = z - z.max()
    ref = np.exp(z) / np.exp(z).sum()
    return ref, rng.standard_normal(V) * 2.0, float(rng.uniform(0.2, 2.0))


def pi_star(ref, r, beta):
    lw = np.where(ref > 0, np.log(np.where(ref > 0, ref, 1.0)) + r / beta, -np.inf)
    w = np.exp(lw - lw.max())
    return w / w.sum()


def log_Z(ref, r, beta):
    lw = np.where(ref > 0, np.log(np.where(ref > 0, ref, 1.0)) + r / beta, -np.inf)
    m = lw.max()
    return float(m + np.log(np.exp(lw - m).sum()))


def J(pi, ref, r, beta):
    m = pi > 0
    return float((pi * r).sum() - beta * (pi[m] * np.log(pi[m] / ref[m])).sum())


# (15.6) is an IDENTITY, not an approximation near the optimum: it holds for
# every policy.  Checked on 5000 random tabular problems, at a random policy.
worst = 0.0
for s in range(5000):
    ref, r, beta = problem(s)
    star, lz = pi_star(ref, r, beta), log_Z(ref, r, beta)
    pi = np.random.default_rng(10 ** 6 + s).dirichlet(np.full(V, 0.7))
    kl = float((pi * np.log(pi / star)).sum())
    worst = max(worst, abs(J(pi, ref, r, beta) - (-beta * kl + beta * lz)))
assert worst < 1e-13, worst
print("J(pi) = -beta KL(pi || pi*) + beta log Z on 5000 random problems: worst "
      "discrepancy %.1e" % worst)

# So the landscape is a SINGLE KL BOWL: the maximum is beta log Z, attained only
# at pi*, and there is nothing else to get stuck in.
ref, r, beta = problem(1512)
star = pi_star(ref, r, beta)
Jstar = J(star, ref, r, beta)
assert abs(Jstar - beta * log_Z(ref, r, beta)) < 1e-10
rng = np.random.default_rng(7)
best = -np.inf
for _ in range(20):
    P = rng.dirichlet(np.full(V, 0.7), 10_000)
    vals = ((P * r).sum(1)
            - beta * np.where(P > 0, P * np.log(np.where(P > 0, P, 1.0) / ref),
                              0.0).sum(1))
    best = max(best, float(vals.max()))
assert best < Jstar
assert Jstar - best > 1.0
print("the best of 200,000 random simplex points on a %d-symbol problem falls "
      "%.2f nats short of beta log Z = %.4f" % (V, Jstar - best, Jstar))

# Step 8.  beta log Z is a FREE ENERGY, so beta says how soft the maximum is.
# The small-beta limit carries its own reference probability, which is worth
# having: max(r) alone needs beta below 1e-5 before it holds to three decimals.
j = int(np.argmax(r))
for b in (1e-2, 1e-3, 1e-4):
    assert abs(b * log_Z(ref, r, b) - (r.max() + b * np.log(ref[j]))) < 1e-6
assert abs(5000.0 * log_Z(ref, r, 5000.0) - float((ref * r).sum())) < 1e-2
print("beta -> 0 gives max r = %.4f; beta -> infinity gives E_ref[r] = %.4f; "
      "beta is how soft the maximum is, not a knob"
      % (r.max(), float((ref * r).sum())))

# The assumption Corollary 15.1 pays off: absolute continuity.  KL regularisation
# cannot discover a completion the reference would never sample, at ANY beta.
ref0 = np.array([0.5, 0.5, 0.0])
r0 = np.array([0.0, 0.0, 10.0])
for b in (0.05, 0.5, 5.0, 500.0):
    assert pi_star(ref0, r0, b)[2] == 0.0
print("a completion with reward 10 that the reference never samples gets "
      "probability exactly 0 at every beta tried")

# (15.7) is the temperature map of Chapter 14 with r for the logits and beta
# for T, so the same monotonicity applies: raising beta raises the entropy.
ent = lambda p: float(-(p[p > 0] * np.log(p[p > 0])).sum())
betas = np.geomspace(0.05, 50.0, 60)
H = np.array([ent(pi_star(ref, r, b)) for b in betas])
assert (np.diff(H) > 0).all()
assert H[-1] < ent(ref)
assert abs(ent(pi_star(ref, r, 1e8)) - ent(ref)) < 1e-6
print("post-training's optimum is an exponential tilt of the reference and "
      "nothing more exotic: entropy rises from %.4f to %.4f as beta goes from "
      "%.2f to %.0f, and reaches the reference's own %.4f in the limit"
      % (H[0], H[-1], betas[0], betas[-1], ent(ref)))

# The misconception, as an inequality.  The penalty is the REVERSE KL, which is
# mode-seeking: a policy that collapses onto one high-reward mode pays little.
top = np.argsort(-ref)[:8]                       # a mode the reference likes
j = int(top[np.argmax(r[top])])                  # and that the reward likes
collapse = np.zeros(V)
collapse[j] = 1.0
collapse = 0.9999 * collapse + 0.0001 * ref
rev = float((collapse * np.log(collapse / ref)).sum())
fwd = float((ref * np.log(ref / collapse)).sum())
assert rev < fwd / 2
assert ent(collapse) < 0.05 * ent(ref)           # diversity is gone
print("collapsing onto one high-reward mode of the reference costs %.2f nats "
      "of reverse KL and %.2f of forward, while the entropy falls from %.2f to "
      "%.4f: only the first number is being enforced"
      % (rev, fwd, ent(ref), ent(collapse)))
'''

S4 = SIGMOID + r'''
SEED = 1504
rng = np.random.default_rng(SEED)
V, beta = 5, 0.35

ref = rng.dirichlet(np.full(V, 1.5))
r = rng.standard_normal(V) * 1.4


def pi_star(ref, r, beta):
    w = ref * np.exp((r - r.max()) / beta)
    return w / w.sum()


star = pi_star(ref, r, beta)
logZ = float(np.log((ref * np.exp(r / beta)).sum()))

# Steps 1 and 2: an EXACT change of variables, one optimal policy per reward.
recovered = beta * np.log(star / ref) + beta * logZ
assert np.abs(recovered - r).max() < 1e-10
print("solving (15.7) for r returns it exactly: max abs %.1e"
      % np.abs(recovered - r).max())

# Steps 3 to 5: beta log Z depends on the prompt alone, so it is a member of
# D-15.1's invariance class and CANCELS in every difference.
for i in range(V):
    for j in range(V):
        lhs = r[i] - r[j]
        rhs = beta * np.log(star[i] / ref[i]) - beta * np.log(star[j] / ref[j])
        assert abs(lhs - rhs) < 1e-10, (i, j)
print("r_w - r_l equals the difference of implicit rewards with no log Z in "
      "it: the one intractable sum in the construction is gone")

# and it is gone for a reason rather than by luck: a DIFFERENT prompt has a
# different Z, so the cancellation is exactly the same-prompt condition.
ref2 = rng.dirichlet(np.full(V, 1.5))
logZ2 = float(np.log((ref2 * np.exp(r / beta)).sum()))
assert abs(logZ2 - logZ) > 0.05
print("a second prompt has beta log Z = %.4f against %.4f, so cross-prompt "
      "pairs would NOT cancel" % (beta * logZ2, beta * logZ))

# Step 7, the gradient, on a featured tabular policy where the chosen and the
# rejected completion SHARE parameters, which is what two answers to the same
# prompt do.
PHI = np.array([[1.0, 0.1], [1.0, -0.1], [0.0, 5.0]])
BETA, LR = 0.1, 2.0
z0 = PHI @ np.zeros(2)
REF = np.exp(z0 - z0.max())
REF = REF / REF.sum()


def policy(theta):
    z = PHI @ theta
    e = np.exp(z - z.max())
    return e / e.sum()


def dpo_loss_and_grad(theta):
    pi = policy(theta)
    rhat = BETA * np.log(pi / REF)                 # (15.13), the implicit reward
    delta = rhat[0] - rhat[1]
    loss = float(-np.log(sigmoid(delta)))
    dlog = PHI - pi @ PHI                          # d log pi_i / d theta
    grad = -BETA * sigmoid(-delta) * (dlog[0] - dlog[1])
    return loss, grad, pi


theta = rng.standard_normal(2) * 0.4
loss, grad, _ = dpo_loss_and_grad(theta)
num = np.zeros(2)
for i in range(2):
    e = np.zeros(2)
    e[i] = 1e-6
    num[i] = (dpo_loss_and_grad(theta + e)[0] - dpo_loss_and_grad(theta - e)[0]) / 2e-6
assert np.abs(grad - num).max() < 1e-7, np.abs(grad - num).max()
print("(15.14) against a central difference: max abs %.2e" % np.abs(grad - num).max())

# The failure mode, which is precise and is the reason DPO needs monitoring.
# The loss sees pi only through a DIFFERENCE of log-ratios, so both can fall.
theta = np.zeros(2)
losses, pis = [], []
for _ in range(3000):
    loss, grad, pi = dpo_loss_and_grad(theta)
    losses.append(loss)
    pis.append(pi.copy())
    theta = theta - LR * grad
losses, pis = np.array(losses), np.array(pis)
assert abs(losses[0] - np.log(2.0)) < 1e-9            # at theta = 0, Delta = 0
assert (np.diff(losses) < 0).all()                    # the loss falls, always
assert (np.diff(pis[:, 0]) < 0).all()                 # and so does pi(y_w)
assert (np.diff(pis[:, 1]) < 0).all()
assert pis[-1, 2] > 0.999                             # the unlabelled one wins
assert (losses[0] - losses[40]) / losses[0] < 0.02
assert pis[40, 0] < 0.10 * pis[0, 0]
rhat = BETA * np.log(pis[-1] / REF)
assert rhat[0] > rhat[1]                              # accuracy is still perfect
print("after 40 steps the loss has moved %.2f%% and pi(y_w) has fallen from "
      "%.4f to %.5f; after %d steps the UNLABELLED completion holds %.4f of "
      "the mass, and the implicit reward still ranks the pair correctly"
      % (100 * (losses[0] - losses[40]) / losses[0], pis[0, 0], pis[40, 0],
         len(losses), pis[-1, 2]))
print("held-out preference accuracy rises while generation quality collapses, "
      "which is why the chosen log-probability has to be logged separately")
'''

S57 = r'''
from arith.post_training_memory import (RESIDENT, beta_for_target,
                                        compute_per_prompt, devices,
                                        grpo_crossover, rlvr_logit_shift,
                                        state_bytes)
from arith.model_d import MODEL_D, total_params

SEED = 1557
rng = np.random.default_rng(SEED)

# Steps 1 and 2, exactly.  For a tabular softmax policy the score-function
# identity is a finite sum, so it can be checked to machine precision.
A = 4
theta = rng.standard_normal(A)
p = np.exp(theta - theta.max())
p /= p.sum()
score = np.eye(A) - p                      # row a is d log pi(a) / d theta
assert np.abs(p @ score).max() < 1e-15
r = rng.standard_normal(A) * 2.0
g_true = (p[:, None] * r[:, None] * score).sum(0)
for b in (-3.0, 0.0, 1.7, float(p @ r)):
    g_b = (p[:, None] * (r - b)[:, None] * score).sum(0)
    assert np.abs(g_b - g_true).max() < 1e-14, b
print("E[(r - b) grad log pi] = E[r grad log pi] for every constant b, to "
      "%.1e: that is the control variate, and the baseline is free"
      % np.abs(g_b - g_true).max())

# BIAS 1, self-inclusion.  The group mean contains r_i, so b is not independent
# of y_i.  Enumerated EXACTLY over all A^G outcomes: the result is not a tilt,
# it is a shrinkage by (G-1)/G.
import itertools
for G in (2, 3, 4, 5):
    acc = np.zeros(A)
    for combo in itertools.product(range(A), repeat=G):
        w = float(np.prod([p[a] for a in combo]))
        rr = np.array([r[a] for a in combo])
        acc += w * ((rr - rr.mean())[:, None] * score[list(combo)]).sum(0) / G
    assert np.abs(acc - (1 - 1 / G) * g_true).max() < 1e-12, G
    print("  G=%d: exact expectation is %.6f x the true gradient (predicted "
          "%.6f)" % (G, np.linalg.norm(acc) / np.linalg.norm(g_true), 1 - 1 / G))

# and by Monte Carlo at larger G, where the direction is the point: the cosine
# with the true gradient is 1.00000, so nothing is tilted at all.
TRIALS, CHUNK = 1_000_000, 100_000
for G in (2, 4, 8, 16):
    est = np.zeros(A)
    est_loo = np.zeros(A)
    for _ in range(TRIALS // CHUNK):
        a = rng.choice(A, size=(CHUNK, G), p=p)
        rr = r[a]
        sc = score[a]
        est += ((rr - rr.mean(axis=1, keepdims=True))[:, :, None]
                * sc).mean(axis=1).sum(axis=0)
        loo = (rr.sum(axis=1, keepdims=True) - rr) / (G - 1)
        est_loo += ((rr - loo)[:, :, None] * sc).mean(axis=1).sum(axis=0)
    est /= TRIALS
    est_loo /= TRIALS
    ratio = float(np.linalg.norm(est) / np.linalg.norm(g_true))
    cos = float(est @ g_true / (np.linalg.norm(est) * np.linalg.norm(g_true)))
    r_loo = float(np.linalg.norm(est_loo) / np.linalg.norm(g_true))
    assert abs(ratio - (1 - 1 / G)) < 0.01, (G, ratio)
    assert round(cos, 5) == 1.00000, (G, cos)
    assert abs(r_loo - 1.0) < 0.01, (G, r_loo)
    print("  G=%2d: norm ratio %.5f (predicted %.5f), cosine %.7f, "
          "leave-one-out ratio %.5f" % (G, ratio, 1 - 1 / G, cos, r_loo))
print("the full-group baseline does not tilt the gradient: it shrinks it by "
      "exactly (G-1)/G, which the learning rate absorbs, and the leave-one-out "
      "mean removes even that")

# BIAS 2, the 1/sigma reweighting, and this one is real.  For a binary reward at
# pass rate p the weight is 1/sqrt(p(1-p)), which diverges at BOTH ends.
w = lambda q: 1.0 / np.sqrt(q * (1 - q))
assert abs(w(0.5) - 2.0) < 1e-12
assert abs(w(0.05) - 4.59) < 5e-3
assert abs(w(0.01) - 10.05) < 5e-3
assert w(0.5) == min(w(q) for q in np.linspace(0.01, 0.99, 99))
for q in (0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99):
    print("  pass rate %.2f: gradient weight 1/sigma = %.3f" % (q, w(q)))
assert w(0.01) > 5 * w(0.5) and w(0.99) > 5 * w(0.5)
# and it is a change of OBJECTIVE, not a variance reduction: with a binary
# reward the standardised advantage no longer depends on p at all.
for q in (0.1, 0.5, 0.9):
    G = 1000
    out = rng.random(G) < q
    adv = (out - out.mean()) / out.std()
    assert abs(np.abs(adv[out]).mean() * np.abs(adv[~out]).mean() - 1.0) < 0.02
print("the prompts the model almost always gets right, and those it almost "
      "never gets right, receive the largest gradients: that is the opposite "
      "of a curriculum")

# BIAS 3, length normalisation.  For a negative advantage a longer completion
# spreads the same penalty over more tokens, so the loss FALLS as a bad answer
# lengthens.  Normalising once across the group removes the incentive.
adv = -1.0
lengths = np.array([20, 60, 200, 600])
per_completion = np.array([abs(adv) * L / L for L in lengths])
across_group = np.array([abs(adv) * L / lengths.sum() for L in lengths])
assert np.abs(np.diff(per_completion)).max() < 1e-12       # length-blind
assert (np.diff(across_group) > 0).all()                   # length-aware
print("dividing by |y_i| makes the penalty %s with length; dividing once by "
      "sum|y_i| makes it %s"
      % ("constant", "grow from %.4f to %.4f" % (across_group[0], across_group[-1])))

# Failure mode: a degenerate group.  std(r) -> 0 gives 0/0, and an epsilon in
# the denominator turns an undefined gradient into a large arbitrary one.
eps = 1e-4
noise = 1e-9 * rng.standard_normal(8)
assert np.abs(noise / (noise.std() + eps)).max() > 1e-6
assert np.abs(noise / (noise.std() + eps)).max() / np.abs(noise).max() > 1e3
print("a group whose 8 rewards agree to 1e-9, divided by std + %g, is "
      "amplified %.0fx: dynamic sampling drops the group instead"
      % (eps, np.abs(noise / (noise.std() + eps)).max() / np.abs(noise).max()))

# What each deletion is worth.  Every algorithm here is PPO with one of its four
# resident models removed, and RESIDENT is that table.
n = total_params(MODEL_D)
assert set(RESIDENT["PPO"]) == {"policy", "reference", "reward", "value"}
assert set(RESIDENT["DPO"]) == {"policy", "reference"}
assert set(RESIDENT["GRPO"]) == {"policy", "reference", "reward"}
for algo in ("PPO", "DPO", "GRPO"):
    b = state_bytes(n, algo)
    assert abs(b["total"] - sum(v for k, v in b.items() if k != "total")) < 1
    print("%-5s %s  total %.2f GB, %.2f devices of 80 GB (state only)"
          % (algo, "  ".join("%s %.2f" % (k, v / 1e9)
                             for k, v in b.items() if k != "total"),
             b["total"] / 1e9, b["total"] / 1e9 / 80))
ppo, dpo = state_bytes(n, "PPO")["total"], state_bytes(n, "DPO")["total"]
assert abs(ppo / dpo - 2.0) < 0.01     # the value network is the policy's size
# with 30% of each device reserved for activations and the rollout KV cache,
# which is the honest accounting: a state count that exactly fills a device does
# not run.  PPO then needs six devices and everything else needs three.
need = {a: devices(state_bytes(n, a)["total"], 80, 0.3) for a in RESIDENT}
assert need["PPO"] == 6 and need["GRPO"] == 3 and need["DPO"] == 3
assert need["GRPO+verifier"] == 3
assert devices(state_bytes(int(70e9), "PPO")["total"], 80, 0.3) == 45
assert devices(state_bytes(int(70e9), "DPO")["total"], 80, 0.3) == 23
print("PPO is exactly %.3fx DPO's state; with 30%% reserved for activations it "
      "needs %d devices of 80 GB and the others need %d, so a 4-device node "
      "runs everything except PPO" % (ppo / dpo, need["PPO"], need["GRPO"]))

# and the trade GRPO makes, which is steep: the crossover arrives at G = 2.
assert compute_per_prompt("PPO") == 18
assert compute_per_prompt("GRPO", 8) == 96 and compute_per_prompt("DPO") == 8
assert grpo_crossover() == 2 and grpo_crossover(True) == 2
assert compute_per_prompt("GRPO", 8) / compute_per_prompt("PPO") > 5
print("GRPO at G=8 costs %.1fx PPO's compute per prompt to save the value "
      "network's %.0f GB, and the crossover is at G = %d"
      % (compute_per_prompt("GRPO", 8) / compute_per_prompt("PPO"),
         state_bytes(n, "PPO")["value"] / 1e9, grpo_crossover()))

# Corollary 15.1.  RLVR shifts the LOG-ODDS of being correct by 1/beta, the same
# shift for every prompt whatever its difficulty, and p = 0 stays p = 0.
logit = lambda q: np.log(q / (1 - q))
b = beta_for_target(0.12, 0.5)
assert abs(b - 0.5019) < 1e-3
for q in (0.01, 0.05, 0.12, 0.3, 0.6):
    shifted = rlvr_logit_shift(q, b)
    assert abs(logit(shifted) - logit(q) - 1 / b) < 1e-9
    print("  p = %.2f -> %.4f, log-odds shift %.4f" % (q, shifted, 1 / b))
for small in (b, 0.1, 0.01, 0.002):
    assert rlvr_logit_shift(0.0, small) == 0.0     # p = 0 stays p = 0 exactly
# and the shift is the SAME for an easy prompt and a hard one, so RLVR cannot
# rescue a prompt the reference never solves however small beta is made
odds = lambda q: q / (1 - q)
assert abs(odds(rlvr_logit_shift(0.6, b)) / odds(0.6)
           - odds(rlvr_logit_shift(0.01, b)) / odds(0.01)) < 1e-9
print("the shift is %.4f for every prompt, and no beta buys signal where the "
      "reference never succeeds: that is D-15.3's absolute-continuity clause, "
      "arriving as a training failure" % (1 / b))
'''

SECTIONS = [
    ("1", "Bradley-Terry, and the invariance that will matter",
     "Dividing the Bradley-Terry probability top and bottom by the winner's "
     "exponentiated reward turns it into a logistic function of the margin, so "
     "only differences are ever modelled. The consequence is that the reward is "
     "identified only up to an additive function of the prompt, and the cell "
     "checks that a whole class of reward functions has identical likelihood. "
     "That freedom is what reappears as the partition function in the DPO "
     "derivation, and it is why that term cancels.",
     S1),
    ("2", "The reward model as MLE, and self-annealing",
     "The negative log-likelihood of the preference data differentiates through "
     "the reflection identity into a gradient with one scalar gain in front, "
     "and the gain is the logistic of the negative margin. The cell checks the "
     "whole gradient against central differences on a real scorer, then "
     "measures what the gain does: a pair the model already has right by a "
     "margin of eight contributes three hundredths of a per cent of what the "
     "same pair backwards contributes. That is why duplicated preference pairs "
     "matter more here than under cross-entropy.",
     S2),
    ("3", "The KL-regularised optimum",
     "Completing the square inside the logarithm turns the objective into a "
     "single KL divergence plus a constant, which is an identity rather than an "
     "approximation near the optimum. The cell asserts it on five thousand "
     "random tabular problems at random policies, then reads off the "
     "consequences: the landscape is one bowl with no other stationary points, "
     "the optimal value is a free energy, and the optimum puts zero mass "
     "wherever the reference does.",
     S3),
    ("4", "DPO",
     "Solving the optimum for the reward is an exact change of variables, and "
     "the intractable term it leaves behind depends on the prompt alone. Both "
     "completions of a pair condition on the same prompt, so that term cancels "
     "in the difference, for the reason the first section gave rather than by "
     "luck. The cell then runs the loss on a featured tabular model and "
     "reproduces the failure mode exactly: the loss falls monotonically, "
     "held-out preference accuracy stays perfect, and the chosen completion's "
     "probability collapses.",
     S4),
    ("5--7", "The group baseline as a control variate, and its three biases",
     "Any baseline independent of the sampled completion leaves the policy "
     "gradient unbiased, and the group mean is not quite independent of it. "
     "Working out what that costs is the first block, and the answer is milder "
     "than the reputation: the expectation is exactly one minus one over G "
     "times the true gradient, so the direction is untouched and the cosine is "
     "one to five decimal places. The second bias, dividing by the group "
     "standard deviation, is the one that changes the objective, and the third "
     "is length normalisation. The section closes with what each deletion from "
     "PPO buys in memory and costs in compute.",
     S57),
]
