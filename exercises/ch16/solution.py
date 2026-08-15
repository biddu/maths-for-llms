"""Your solutions for Chapter 16's [C] exercises.

THIS IS THE `solutions` BRANCH.  Every function below is worked in full and
every test in this directory passes.  On `main` these are stubs raising
NotImplementedError, and making them pass is the exercise; read this file only
after you have had a go, or when you want to compare a method rather than an
answer.  Appendix C prints the answer and the tolerance, not the code.
"""
import numpy as np


# ------------------------------------------------------------------ E-16.9
def soft_threshold(c, lam):
    """(16.17).  sign(c) max(|c| - lam/2, 0), elementwise.

    The threshold is lam/2 and not lam.  The factor of two comes from
    differentiating the squared reconstruction term, and getting it wrong is the
    single most common slip in this derivation.
    """
    c = np.asarray(c, dtype=np.float64)
    return np.sign(c) * np.maximum(np.abs(c) - lam / 2.0, 0.0)


def topk(c, k):
    """Keep the k largest-magnitude coordinates of c UNCHANGED, zero the rest.

    Unchanged is the point: TopK has no penalty term, so stationarity gives
    z_j = c_j on the chosen set and there is no shrinkage to correct.
    """
    c = np.asarray(c, dtype=np.float64)
    z = np.zeros_like(c)
    keep = np.argpartition(-np.abs(c), k - 1)[:k]
    z[keep] = c[keep]
    return z


def reconstruction_ratio(c_active, lam):
    """||xhat|| / ||x_parallel|| under soft-thresholding, for active atoms.

    With every atom at the same magnitude cbar this is 1 - lam/(2 cbar), which
    does NOT depend on how many atoms are active.  That independence is why the
    bias cannot be tuned away with a sparsity sweep.

    Method note.  With near-orthogonal atoms the two norms are the norms of the
    coefficient vectors, so the ratio is ||soft_threshold(c)|| / ||c||.  Written
    that way it holds for any active set; at a common magnitude cbar it reduces
    to arith.sae_capacity.soft_threshold_deficit(lam, cbar), because every
    coordinate loses the same lam/2 and the sqrt(k) cancels.
    """
    c = np.asarray(c_active, dtype=np.float64)
    return float(np.linalg.norm(soft_threshold(c, lam)) / np.linalg.norm(c))


# ----------------------------------------------------------------- E-16.10
def random_dictionary(m, d, rng):
    """m unit vectors drawn uniformly on the sphere in R^d, shape (m, d)."""
    U = rng.standard_normal((int(m), int(d)))
    return U / np.linalg.norm(U, axis=1, keepdims=True)


def max_coherence(U):
    """The largest |cos| between distinct rows of U.

    Build the Gram matrix, blank the diagonal, take the largest absolute entry.
    At m = 20000 the Gram matrix is 3.2 GB in float64, so chunk it or use
    float32; a solution that allocates it whole is not wrong, it just will not
    run at the sizes the exercise asks for.
    """
    U = np.asarray(U, dtype=np.float64)
    m = U.shape[0]
    if m < 2:
        return 0.0
    best = 0.0
    chunk = max(1, int(2 ** 24 // max(m, 1)))    # one chunk of Gram at a time
    for s in range(0, m, chunk):
        G = np.abs(U[s:s + chunk] @ U.T)
        rows = np.arange(s, min(s + chunk, m))
        G[rows - s, rows] = 0.0                  # blank the diagonal
        best = max(best, float(G.max()))
    return best


def largest_m_within(eps, d, rng, hi=4096):
    """The largest m (by bisection, up to hi) whose random draw has coherence
    at most eps.  Compare with exp(d eps^2 / 4) and report the ratio: that ratio
    is how loose the union bound of D-16.2 is, and measuring it is the exercise.

    Method note.  Coherence grows with m only in distribution, so each probe
    draws a fresh dictionary and the bisection is on a noisy predicate.  That is
    honest for this measurement: what is being estimated is the m at which a
    typical draw crosses eps, not a quantity any single dictionary owns.
    """
    lo, high = 1, int(hi)
    if max_coherence(random_dictionary(high, d, rng)) <= eps:
        return high
    while lo < high:
        mid = (lo + high + 1) // 2
        if max_coherence(random_dictionary(mid, d, rng)) <= eps:
            lo = mid
        else:
            high = mid - 1
    return lo


# ----------------------------------------------------------------- E-16.11
def _reads(U, k, n_trials, rng):
    """Matched-filter reads, pooled over trials, split active from inactive.

    Each trial lights k features at unit intensity, forms x as their sum and
    reads every feature with its own atom, which is the decoder-transpose read
    of D-16.3.
    """
    U = np.asarray(U, dtype=np.float64)
    m = U.shape[0]
    active, inactive = [], []
    for _ in range(int(n_trials)):
        idx = rng.choice(m, size=int(k), replace=False)
        r = U @ U[idx].sum(axis=0)
        on = np.zeros(m, dtype=bool)
        on[idx] = True
        active.append(r[on])
        inactive.append(r[~on])
    return np.concatenate(active), np.concatenate(inactive)


def interference_std(U, k, n_trials, rng):
    """Standard deviation of the matched-filter read at INACTIVE features.

    Draw k active features with unit intensities, form x, read every feature,
    and take the standard deviation over the inactive ones.  D-16.3 step 6 says
    this is sqrt(k/d) for a random dictionary; confirming that is the point.
    """
    _, inactive = _reads(U, k, n_trials, rng)
    return float(inactive.std())


def largest_k_separating(U, z_sigma, n_trials, rng, kmax=1024):
    """The largest k at which active and inactive reads are separated by at
    least z_sigma standard deviations of the interference.  Compare with
    d / z_sigma**2, and with (16.14) at z_sigma**2 = 4 ln m.

    Method note.  The active read sits at its own intensity, 1, and the
    inactive read at 0, so the gap to be paid for is 1 and the price is
    z_sigma sqrt(k/d).  Solving 1 >= z_sigma sqrt(k/d) gives k <= d/z_sigma^2,
    which at z_sigma^2 = 4 ln m is exactly (16.14).  The predicate is monotone
    in k because the interference grows as sqrt(k), so bisection is safe.
    """
    def separated(k):
        active, inactive = _reads(U, k, n_trials, rng)
        return (active.mean() - inactive.mean()) >= z_sigma * inactive.std()

    lo, high = 1, min(int(kmax), U.shape[0] - 1)
    if separated(high):
        return high
    while lo < high:
        mid = (lo + high + 1) // 2
        if separated(mid):
            lo = mid
        else:
            high = mid - 1
    return lo


# ----------------------------------------------------------------- E-16.12
def splitting_fraction(U_small, U_large, thresh=0.7):
    """Fraction of rows of U_small matched by two or more rows of U_large above
    `thresh` in absolute cosine.

    This is the quantity people report as evidence of feature splitting.  The
    exercise asks you to compute it and then to say why it cannot settle how
    many features the model has: it is a property of the two dictionaries you
    happened to fit, and nothing in the objective picks a width.
    """
    A = np.asarray(U_small, dtype=np.float64)
    B = np.asarray(U_large, dtype=np.float64)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    matches = (np.abs(A @ B.T) > thresh).sum(axis=1)
    return float(np.mean(matches >= 2))
