"""Chapter 12 — Mixture of Experts.

Generated into `notebooks/ch12_moe.ipynb` by `build_all.py`.  The chapter cites
§1, §2 and §4 by number, so sections may be added but never renumbered.

Two places where the notebook agrees with the corrected chapter rather than
with the obvious argument.  §1: the unselected gradient vanishes because the
renormalised gate is homogeneous of degree zero, so the contraction of the
Jacobian AGAINST THE GATES is zero; the sum of the Jacobian's entries is not,
and the cell measures it at about 0.51 to make the difference visible.  §4: the
limit-cycle ripple has amplitude of order u times g_p, not u divided by it.
"""
from __future__ import annotations

CHAPTER = 12
SLUG = "moe"
TITLE = "Mixture of Experts"
BLURB = (
    "A hard choice with a differentiable weight attached, and the three "
    "consequences: no gradient reaches an unselected expert, the auxiliary "
    "loss is a variance penalty in disguise, and the bias controller is an "
    "integrator whose ripple is the price of its gain."
)

S1 = r'''
SEED = 12
rng = np.random.default_rng(SEED)
E, k, d = 16, 4, 32

x = rng.normal(size=d)
W_r = rng.normal(size=(d, E)) / np.sqrt(d)
experts = rng.normal(size=(E, d))              # E_i(x), held fixed: step 3's assumption

z = x @ W_r
g = np.exp(z - z.max()); g /= g.sum()
T = np.argsort(-z)[:k]                          # the selected set
S = g[T].sum()
unselected = [m for m in range(E) if m not in set(T.tolist())]
assert len(unselected) == E - k

# ---- step 1: the selection is piecewise constant, so away from ties it
# contributes nothing.  Checked by perturbing every logit by a small amount and
# confirming the selected set never moves.
gap = z[T].min() - max(z[m] for m in unselected)
assert gap > 1e-3, gap                          # no tie at the boundary
for m in range(E):
    for h in (-gap / 4, gap / 4):
        zz = z.copy(); zz[m] += h
        assert set(np.argsort(-zz)[:k].tolist()) == set(T.tolist()), (m, h)
print("the top-%d set is stable under perturbations of every logit; the "
      "boundary gap is %.4f" % (k, gap))

# ---- step 4: the softmax Jacobian is NOT zero off the selection.  Softmax
# couples every logit to every other, and that is what makes the result
# surprising enough to need a derivation.
for m in unselected:
    for j in T:
        dgj_dzm = -g[j] * g[m]
        assert abs(dgj_dzm) > 1e-8, (m, j)
print("dg_j/dz_m = -g_j g_m is never zero: the largest off-selection coupling "
      "is %.3e" % max(abs(-g[j] * g[m]) for m in unselected for j in T))

# ---- step 6, and this is the whole derivation.  g_hat is homogeneous of
# degree zero in the SELECTED gates, so by Euler's theorem the contraction of
# its Jacobian against the gates vanishes.
J = np.array([[((1.0 if i == j else 0.0) * S - g[T][i]) / S ** 2
               for j in range(k)] for i in range(k)])
euler = J @ g[T]                                # sum_j g_j dg_hat_i/dg_j
assert np.abs(euler).max() < 1e-15, np.abs(euler).max()
# and directly, as the definition of degree-zero homogeneity: multiply every
# selected gate by a common factor and g_hat does not move
ghat = g[T] / S
for scale in (0.5, 1.0, 2.0, 100.0):
    assert np.abs((scale * g[T]) / (scale * g[T]).sum() - ghat).max() < 1e-15
print("Euler contraction sum_j g_j dg_hat_i/dg_j = %.1e for every i, and "
      "g_hat(c g) = g_hat(g) exactly" % np.abs(euler).max())

# ---- the quantity that is NOT what closes the argument, measured.  Summing
# the Jacobian's entries is a different contraction, it is true of nothing in
# particular, and on this router it is about a half.
naive = float(sum(g[T][i] * (S - g[T][i]) / S ** 2 for i in range(k)))
# it is the DIAGONAL of the same Jacobian contracted with the gates, where
# Euler's theorem is about the full rows, and the two are not the same number
assert abs(naive - float(np.diag(J) @ g[T])) < 1e-15
assert abs(np.abs(euler).max() - 0.0) < 1e-15 and abs(naive) > 0.1
assert abs(naive - 0.514) < 0.01, naive
print("diag(J) . g = %.4f, which does NOT vanish, while J g = %.1e, which "
      "does: the two contractions differ and only the second closes the "
      "argument" % (naive, np.abs(euler).max()))

# ---- step 7, as a measurement.  Renormalisation is what kills the gradient,
# so the same finite difference is run with it and without it.
def out(zz, renorm):
    gg = np.exp(zz - zz.max()); gg /= gg.sum()
    gh = gg[T] / gg[T].sum() if renorm else gg[T]
    return (gh[:, None] * experts[T]).sum(0)


h = 1e-7


def slope(m, renorm):
    up, dn = z.copy(), z.copy()
    up[m] += h; dn[m] -= h
    return np.abs(out(up, renorm) - out(dn, renorm)).max() / (2 * h)


with_renorm = max(slope(m, True) for m in unselected)
without = max(slope(m, False) for m in unselected)
selected = max(slope(m, True) for m in T)
assert with_renorm < 1e-8, with_renorm            # zero, to the difference's noise
assert 1e-9 < with_renorm < 5e-9                  # about 2e-9 on this router
assert without > 1e-2 and abs(without - 0.056) < 0.005
assert selected > 0.1                             # and the selected ones do move
assert without / with_renorm > 1e6
print("d|y|/dz on an unselected logit: %.1e with renormalisation, %.1e "
      "without, a factor of %.0e. Selected logits: %.3f"
      % (with_renorm, without, without / with_renorm, selected))

# ---- the failure mode, which is the organising idea of the chapter.  An
# expert that is never selected receives nothing, so it never improves, so it
# is never selected: a k-armed bandit played by a greedy policy.
import os


def repo_file(*parts):
    for base in ("..", "."):
        q = os.path.join(base, *parts)
        if os.path.exists(q):
            return q
    raise FileNotFoundError(os.path.join(*parts))


Z = np.load(repo_file("figs", "data", "moe_regimes.npz"))
loads = Z["none_loads"]                          # three seeds, 64 experts
E_meas = loads.shape[1]
uniform = 1.0 / E_meas
ratio = (loads.max(1) / np.maximum(loads.min(1), 1e-12)).mean()
starved = (loads < uniform / 10).sum(1).mean()
assert E_meas == 64
assert round(ratio, 1) == 29.7
assert round(starved, 1) == 1.3
assert (loads.sum(1) - 1.0).max() < 1e-9         # they are token fractions
print("measured on a %d-expert router with nothing to prevent it: the busiest "
      "expert takes %.1fx the load of the quietest, and %.1f experts on "
      "average fall below a tenth of the uniform share"
      % (E_meas, ratio, starved))
'''

S2 = r'''
import os

SEED = 12002
rng = np.random.default_rng(SEED)
E, T, k = 64, 4096, 6
alpha = 0.001

z = rng.normal(size=(T, E)) * 1.5
g = np.exp(z - z.max(1, keepdims=True)); g /= g.sum(1, keepdims=True)
sel = np.argsort(-z, axis=1)[:, :k]
f = np.bincount(sel.ravel(), minlength=E) / (T * k)
P = g.mean(0)

# ---- step 1: both vectors sum to one, by construction and not by luck.
assert abs(f.sum() - 1) < 1e-12 and abs(P.sum() - 1) < 1e-12
assert (f >= 0).all() and (P > 0).all()

# ---- steps 2 to 6: at P = f the loss is alpha plus alpha E^2 times the
# variance of the load, which is why it penalises imbalance at all.
lhs = alpha * E * float(f @ f)
rhs = alpha + alpha * E ** 2 * float(np.var(f))
assert abs(lhs - rhs) < 1e-15, (lhs, rhs)
# and the rearrangement it comes from, on its own
assert abs(float((f * f).sum()) - (E * np.var(f) + 1.0 / E)) < 1e-15
print("at P = f: alpha E sum f^2 = %.8f and alpha + alpha E^2 Var(f) = %.8f"
      % (lhs, rhs))

# ---- step 7: the minimum is alpha and it does NOT depend on E, which is why
# a value of alpha transfers between models with different expert counts.
for e in (8, 16, 64, 256, 1024):
    u = np.full(e, 1.0 / e)
    assert abs(alpha * e * float(u @ u) - alpha) < 1e-18
    assert np.var(u) == 0.0
# and anything else is strictly larger
for _ in range(200):
    q = rng.dirichlet(np.ones(E))
    assert alpha * E * float(q @ q) > alpha - 1e-15
    if np.var(q) > 1e-9:
        assert alpha * E * float(q @ q) > alpha + 1e-12
print("the minimum is alpha = %.4f at every expert count from 8 to 1024, and "
      "the measured load above sits at %.4f" % (alpha, lhs))

# ---- step 8, the gradient, which is the mechanism.  f is a count of
# assignments, hence piecewise constant, hence carries no gradient at all; the
# whole of it flows through P.
ana = (alpha * E / T) * g * (f[None, :] - (g * f[None, :]).sum(1, keepdims=True))


def aux_loss(zz):
    gg = np.exp(zz - zz.max(1, keepdims=True)); gg /= gg.sum(1, keepdims=True)
    return alpha * E * float(f @ gg.mean(0))


h = 1e-6
for t, j in ((0, 0), (5, 3), (100, 40), (T - 1, E - 1)):
    up, dn = z.copy(), z.copy()
    up[t, j] += h; dn[t, j] -= h
    fd = (aux_loss(up) - aux_loss(dn)) / (2 * h)
    assert abs(fd - ana[t, j]) < 1e-12, (t, j, fd, ana[t, j])
worst = 0.0
for t, j in ((0, 0), (5, 3), (100, 40), (T - 1, E - 1)):
    up, dn = z.copy(), z.copy()
    up[t, j] += h; dn[t, j] -= h
    worst = max(worst, abs((aux_loss(up) - aux_loss(dn)) / (2 * h) - ana[t, j]))
print("equation (12.6) against central differences at four positions: max abs "
      "%.2e" % worst)

# and f really is piecewise constant: perturb a logit and the assignment counts
# do not move, so no gradient can flow through them
for t, j in ((0, 0), (17, 5)):
    zz = z.copy(); zz[t, j] += 1e-6
    f2 = np.bincount(np.argsort(-zz, axis=1)[:, :k].ravel(), minlength=E) / (T * k)
    assert np.array_equal(f, f2), (t, j)
print("f is unchanged by a 1e-6 perturbation of any logit, so it carries no "
      "gradient and every term in (12.6) came through P")

# ---- the sign rule the failure-mode paragraph reads off: expert j's logit is
# pushed DOWN exactly when f_j exceeds this token's probability-weighted mean
# load.  Overloaded experts are discouraged, and only for the tokens that were
# leaning towards them.
weighted_mean = (g * f[None, :]).sum(1)
for t in (0, 11, 222):
    for j in range(E):
        if f[j] > weighted_mean[t] + 1e-9:
            assert ana[t, j] > 0                  # gradient positive = logit down
        elif f[j] < weighted_mean[t] - 1e-9:
            assert ana[t, j] < 0
print("the gradient's sign is set by f_j against <f, g_t> at every position "
      "checked, which is the mechanism and also the defect")

# ---- and the cost, measured.  The auxiliary loss balances by adding a term to
# the objective, and the objective pays for it.  Three regimes, same
# initialisation, same data, same experts.
def repo_file(*parts):
    for base in ("..", "."):
        q = os.path.join(base, *parts)
        if os.path.exists(q):
            return q
    raise FileNotFoundError(os.path.join(*parts))


Z = np.load(repo_file("figs", "data", "moe_regimes.npz"))
cv = {key: float((Z[key + "_loads"].std(1) / Z[key + "_loads"].mean(1)).mean())
      for key in ("none", "aux", "bias")}
lm = {key: float(Z[key + "_loss"].mean()) for key in ("none", "aux", "bias")}
assert round(cv["none"], 2) == 0.55
assert round(cv["aux"], 3) == 0.074
assert round(cv["bias"], 3) == 0.023
# the auxiliary loss does balance, and it costs the objective to do it
assert cv["aux"] < cv["none"] / 5
assert lm["aux"] > lm["none"]
# and the controller of §4 balances better AND costs less, which is the only
# evidence in the chapter that one mechanism dominates another
assert cv["bias"] < cv["aux"] and lm["bias"] < lm["aux"]
print("%-8s %-14s %-12s" % ("regime", "load CV", "final loss"))
for key in ("none", "aux", "bias"):
    print("%-8s %-14.4f %-12.3f" % (key, cv[key], lm[key]))
'''

S3 = r'''
from arith.model_s import MODEL_S, dropped_fraction, expert_capacity

SEED = 12003
rng = np.random.default_rng(SEED)
c = MODEL_S

# ---- (12.7), and the k that secondary sources drop.  With top-k routing there
# are kT assignments to distribute over E experts, not T.
T_batch = 8192
r = expert_capacity(T_batch, c.E, c.k, 1.25)
assert r["mean_load"] == c.k * T_batch / c.E == 256.0
assert r["capacity"] == 320.0
assert abs(r["slack_fraction"] - 0.2) < 1e-12
# dropping the k under-sizes every buffer by a factor of k, which here is eight
without_k = T_batch / c.E
assert without_k == 32.0 and r["mean_load"] / without_k == c.k == 8
assert c.k * T_batch == 65_536                 # assignments, not tokens
print("T = %d tokens, E = %d experts, k = %d: %d assignments, mean load %.0f, "
      "capacity %.0f at c = 1.25. Without the k it would be %.0f, %dx too small"
      % (T_batch, c.E, c.k, c.k * T_batch, r["mean_load"], r["capacity"],
         without_k, c.k))

# the slack is 1 - 1/c of the buffer, and it is paid for in idle memory
for cf in (1.0, 1.25, 1.5, 2.0):
    q = expert_capacity(T_batch, c.E, c.k, cf)
    assert abs(q["slack_fraction"] - (1 - 1 / cf)) < 1e-12
    assert abs(q["wasted_slots"] - (cf - 1) * q["mean_load"] * c.E) < 1e-9
assert expert_capacity(T_batch, c.E, c.k, 2.0)["slack_fraction"] == 0.5
print("at c = 2 half the buffer is never written: %.0f slots per micro-batch"
      % expert_capacity(T_batch, c.E, c.k, 2.0)["wasted_slots"])

# ---- at c = 1 the buffer holds exactly the mean, so ANY imbalance drops
# tokens.  A perfectly balanced load drops none, which is the only case.
balanced = np.full(c.E, r["mean_load"])
assert dropped_fraction(balanced, r["mean_load"]) == 0.0
for jitter in (0.02, 0.1, 0.4):
    loads = balanced * (1 + jitter * rng.standard_normal(c.E))
    loads = np.clip(loads, 0, None) * balanced.sum() / max(loads.sum(), 1e-9)
    d1 = dropped_fraction(loads, r["mean_load"])
    d125 = dropped_fraction(loads, r["capacity"])
    assert d1 > 0.0 and d125 <= d1
    print("jitter %.2f: %.3f%% of assignments dropped at c = 1, %.3f%% at "
          "c = 1.25" % (jitter, 100 * d1, 100 * d125))

# ---- and on the measured loads of F-12.2 rather than on synthetic jitter.
# The unbalanced router drops tokens at a capacity factor the balanced ones sit
# comfortably inside.
import os


def repo_file(*parts):
    for base in ("..", "."):
        q = os.path.join(base, *parts)
        if os.path.exists(q):
            return q
    raise FileNotFoundError(os.path.join(*parts))


Z = np.load(repo_file("figs", "data", "moe_regimes.npz"))
E_meas = Z["none_loads"].shape[1]
mean_load = 1.0 / E_meas                          # loads are token fractions
drops = {}
for key in ("none", "aux", "bias"):
    L = Z[key + "_loads"]                         # one row per seed
    drops[key] = {cf: float(np.mean([dropped_fraction(L[i], cf * mean_load)
                                     for i in range(L.shape[0])]))
                  for cf in (1.0, 1.25, 1.5)}
    assert drops[key][1.0] >= drops[key][1.25] >= drops[key][1.5]
# at c = 1 every regime drops something, because no measured load is exactly
# uniform and c = 1 leaves no room at all
assert all(drops[key][1.0] > 0 for key in drops)
assert drops["none"][1.25] > 0.1                  # the collapsed router drops
assert drops["bias"][1.25] == 0.0                 # the controlled one does not
assert drops["aux"][1.25] == 0.0
assert drops["none"][1.5] > 0.05                  # and still does at c = 1.5
print("%-8s %-12s %-12s %-12s" % ("regime", "c = 1", "c = 1.25", "c = 1.5"))
for key in ("none", "aux", "bias"):
    print("%-8s %-12.4f %-12.4f %-12.4f"
          % (key, drops[key][1.0], drops[key][1.25], drops[key][1.5]))

# ---- and what the slack costs, as the memory line item the chapter says it
# is: b s d p_b (c - 1) bytes per expert per layer.
b, s, p_b = 1, 4096, 2
waste = b * s * c.d * p_b * (1.25 - 1)
assert waste == 1 * 4096 * 7168 * 2 * 0.25
assert round(waste / 1e6, 1) == 14.7
print("at c = 1.25 the slack is %.1f MB per expert per layer, which belongs in "
      "the config comment beside the value" % (waste / 1e6))
'''

S4 = r'''
import os

SEED = 31
E, k, d, N = 32, 4, 64, 4096
rng = np.random.default_rng(SEED)

# A fixed population of tokens through a fixed router, so the only thing moving
# is the bias.  That isolates the controller from the sampling noise a real
# training loop adds, which F-12.3 shows separately.
W_r = rng.normal(size=(d, E)) / np.sqrt(d)
Z = rng.normal(size=(N, d)) @ W_r


def loads(gamma):
    sel = np.argpartition(-(Z + gamma), k, axis=1)[:, :k]
    return np.bincount(sel.ravel(), minlength=E) / (N * k)


# ---- step 1.  The bias is outside the autograd graph by construction: the
# selection is piecewise constant in gamma, and the GATES never see gamma at
# all, so the contribution to the language-modelling loss is identically zero.
def gate(gamma):
    "The gate is a softmax of z ALONE, so gamma cannot appear in it."
    e = np.exp(Z - Z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


base_gates = gate(np.zeros(E))
for shift in (0.3, -1.7, 3.7):
    gamma = np.full(E, shift)
    assert np.abs(gate(gamma) - base_gates).max() == 0.0
    assert np.array_equal(loads(gamma), loads(np.zeros(E)))   # a common shift is a no-op
# a non-uniform bias does move the selection, which is the point of having one
skew = np.zeros(E); skew[0] = 1.0
assert not np.array_equal(loads(skew), loads(np.zeros(E)))
assert np.abs(gate(skew) - base_gates).max() == 0.0
print("the gates never see gamma at all, and a common shift of every bias "
      "changes nothing: only differences matter, which is the zero mode")

# ---- step 4.  Linearise the plant: g_p = d c_i / d gamma_i, positive because
# raising an expert's bias can only make it more likely to be selected.
h = 0.02
gains = []
for i in range(E):
    e_i = np.zeros(E); e_i[i] = h
    up, dn = loads(e_i)[i], loads(-e_i)[i]
    assert up >= dn                                  # monotone, which is step 4
    gains.append((up - dn) / (2 * h))
g_p = float(np.mean(gains))
assert 0.03 < g_p < 0.08, g_p
print("plant gain g_p = %.5f, so the linearised stability band 0 < u g_p < 2 "
      "is u < %.1f and monotone below u < %.1f" % (g_p, 2 / g_p, 1 / g_p))

# ---- steps 3 and 5.  The unrolled recursion is a discrete integrator on load
# error, and near balance it is the linear map delta -> (1 - u g_p) delta.
def linear_orbit(u, steps=200, d0=0.02):
    x = d0
    out = [x]
    for _ in range(steps):
        x = (1 - u * g_p) * x
        out.append(x)
    return np.array(out)


for u in (0.5 / g_p, 1.0 / g_p, 1.5 / g_p, 1.99 / g_p):
    orb = linear_orbit(u)
    assert abs(orb[-1]) < abs(orb[0])                # stable for 0 < u g_p < 2
for u in (2.01 / g_p, 3.0 / g_p):
    assert abs(linear_orbit(u)[-1]) > abs(linear_orbit(u)[0])   # and not beyond
mono = linear_orbit(0.5 / g_p)
alt = linear_orbit(1.5 / g_p)
assert (np.diff(np.sign(mono)) == 0).all()           # monotone when u g_p < 1
assert (np.abs(np.diff(np.sign(alt))) > 0).any()     # alternating when it is not
print("linearised: stable for u g_p in (0, 2), monotone below 1 and "
      "alternating above it, both checked over 200 steps")

# ---- step 7, and this is the correction the chapter carries.  Put the sign
# back and the step has fixed magnitude u, so the loop cannot converge to a
# point: it limit-cycles, and the ripple is of order u TIMES g_p.
def run(u, steps, sign=True):
    gamma = np.zeros(E)
    hist = []
    for _ in range(steps):
        e = 1.0 / E - loads(gamma)
        hist.append(np.abs(e).max())
        gamma = gamma + u * (np.sign(e) if sign else e)
    return np.array(hist)


tails = {}
for u in (0.03, 0.1, 0.3):
    tails[u] = float(run(u, 900)[-200:].mean())
    assert 0.6 < tails[u] / (u * g_p) < 1.7, (u, tails[u], u * g_p)
    # and NOT u / g_p, which is not even dimensionally possible: it is the
    # ratio of a bias to a load, and it comes out two to three orders of
    # magnitude too large
    assert tails[u] / (u / g_p) < 0.01
assert tails[0.03] < tails[0.1] < tails[0.3]
print("%-8s %-12s %-12s %-12s" % ("u", "ripple", "u * g_p", "u / g_p"))
for u in (0.03, 0.1, 0.3):
    print("%-8.2f %-12.5f %-12.5f %-12.3f" % (u, tails[u], u * g_p, u / g_p))
print("the ripple tracks u * g_p to within a factor of 1.2; u / g_p is wrong "
      "by more than a hundredfold, and is a bias where a load is wanted")

# ---- and the proportional form does not diverge past the band either, because
# top-k selection saturates: a plant that cannot deliver more than one hundred
# per cent of the tokens has no unbounded response.
inside = run(0.5 / g_p, 300, sign=False)[-50:].mean()
edge = run(1.85 / g_p, 300, sign=False)[-50:].mean()
beyond = run(2.3 / g_p, 300, sign=False)[-50:].mean()
assert inside < edge < beyond
assert beyond < 0.5, "a saturating plant cannot diverge"
print("proportional control: error %.5f inside the band, %.5f at its edge, "
      "%.5f beyond it, and bounded throughout" % (inside, edge, beyond))

# ---- the measurement behind F-12.3, from the committed file.  Three gains on
# the 64-expert router: the gain buys speed and pays in ripple, and the largest
# one never improves on its own limit cycle.
def repo_file(*parts):
    for base in ("..", "."):
        q = os.path.join(base, *parts)
        if os.path.exists(q):
            return q
    raise FileNotFoundError(os.path.join(*parts))


M = np.load(repo_file("figs", "data", "moe_gains.npz"))
err, gains_meas = M["err"], M["gains"]
assert list(np.round(gains_meas, 6)) == [0.001, 0.01, 0.1]
tail_stats = [(float(err[i, -200:].min()), float(err[i, -200:].mean()),
               float(err[i, -200:].max())) for i in range(3)]
# the ripple grows with the gain, in all three of floor, mean and ceiling
assert tail_stats[0][1] < tail_stats[1][1] < tail_stats[2][1]
assert tail_stats[0][2] < tail_stats[1][2] < tail_stats[2][2]
# the chapter's two quoted rows
assert round(tail_stats[2][1], 4) == 0.0117
assert round(tail_stats[2][0], 4) == 0.0071 and round(tail_stats[2][2], 4) == 0.0203
assert round(tail_stats[0][1], 4) == 0.0014
# and the largest gain never once, at any point in the whole run, gets below
# the smallest gain's steady state: its residual is a limit cycle and not a
# converged state, which is the distinction the caption insists on.
assert err[2].min() > tail_stats[0][1]
every = int(M["every"])


def first_below(row, target):
    ix = np.flatnonzero(row < target)
    return int(ix[0]) * every if len(ix) else None


# the gain buys speed and pays in ripple: the middle gain reaches a moderate
# target first, and the largest never reaches it at all.
assert first_below(err[1], 0.005) < first_below(err[0], 0.005)
assert first_below(err[2], 0.005) is None
# and the smallest gain, given time, goes furthest
assert first_below(err[0], 0.002) is not None
assert first_below(err[1], 0.002) is None
print("%-8s %-10s %-10s %-10s %-14s" % ("u", "tail min", "tail mean",
                                        "tail max", "step to 0.005"))
for i, u in enumerate(gains_meas):
    reached = first_below(err[i], 0.005)
    print("%-8g %-10.4f %-10.4f %-10.4f %-14s"
          % (u, *tail_stats[i], reached if reached is not None else "never"))
'''

SECTIONS = [
    ("1", "The routing gradient sees only the chosen experts",
     "The surprise is that the softmax genuinely couples every logit to every "
     "other and yet no gradient reaches an unselected expert. The cell shows "
     "the coupling is real, then shows what actually kills the gradient: the "
     "renormalised gate is homogeneous of degree zero in the selected gates, so "
     "by Euler's theorem the contraction of its Jacobian against those gates "
     "vanishes. That is not the same as the sum of the Jacobian's entries "
     "vanishing, which is a different quantity and measures about a half here, "
     "and the cell asserts both. Removing the renormalisation puts the "
     "gradient back, seven orders of magnitude larger.",
     S1),
    ("2", "The auxiliary loss is a variance surrogate",
     "The inner product of two vectors that both sum to one is not obviously a "
     "balance penalty, and one rearrangement makes it one: at P equal to f it "
     "is the coefficient plus the coefficient times the square of the expert "
     "count times the variance of the load. The minimum is the coefficient "
     "itself and does not depend on the expert count, which is why a value "
     "transfers between models. The cell then checks the gradient against "
     "central differences, confirms that the assignment counts carry no "
     "gradient at all, and closes with the measured cost: the auxiliary loss "
     "balances the router and the objective pays for it.",
     S2),
    ("3", "Capacity, dropping, and the compute buffer",
     "Balance is a statistical property and a kernel needs a fixed-size buffer, "
     "so each expert gets room for a capacity factor times the mean load, and "
     "the mean load carries a k that secondary sources routinely drop. The cell "
     "computes both, sizes the error at a factor of eight for Model S, and then "
     "measures dropped fractions on the committed per-expert loads: at a "
     "capacity factor of 1.25 the collapsed router loses a real share of its "
     "assignments while the controlled one loses none.",
     S3),
    ("4", "Bias-adjusted routing is integral control",
     "The bias moves who gets chosen and never touches what they are weighted "
     "by, so its contribution to the objective is identically zero rather than "
     "detached, and the cell checks that by shifting every bias and watching "
     "nothing move. Unrolled, the update is a discrete integrator on load "
     "error, and near balance it is a linear map that is stable inside a band "
     "and monotone inside half of it. With the sign function put back the step "
     "has fixed magnitude, so the loop limit-cycles rather than converging, and "
     "the ripple has amplitude of order u times the plant gain, not u divided "
     "by it, which would be a bias where a load is wanted.",
     S4),
]
