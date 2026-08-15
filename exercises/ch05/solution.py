"""Your solutions for Chapter 5's [C] exercises."""

import numpy as np


def rho_per_layer(activations):
    """E-5.10.  |mean(x)|/RMS(x) over the feature axis, per layer.

    `activations` is one array per layer, each (n, d) or (..., d): the ratio is
    formed along the last axis, so the result for a layer is one number per
    token.  The denominator is the RMS itself and not the standard deviation,
    which is the whole point: rho compares the mass a layer puts on the
    all-ones direction with the mass it puts on the sphere, and a small rho is
    what makes centring the removable half of LayerNorm.
    """
    rho = []
    for a in activations:
        a = np.asarray(a, dtype=float)
        mean = a.mean(axis=-1)
        rms = np.sqrt((a * a).mean(axis=-1))
        rho.append(np.abs(mean) / rms)
    return rho


def rmsnorm_forward(x, g, eps=1e-5):
    """E-5.11.  RMSNorm forward, equation (5.4).

    Normalise along the last axis, so x may be a single vector or a batch of
    rows and g is always (d,).
    """
    x = np.asarray(x, dtype=float)
    g = np.asarray(g, dtype=float)
    r = np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    return (x / r) * g


def rmsnorm_backward(x, g, dy, eps=1e-5):
    """E-5.11.  The Jacobian of D-5.1, applied to dy. Return (dx, dg).

    With r the RMS of the row and xhat = x / r,

        dg   = sum over rows of dy * xhat,
        dx_i = g_i dy_i / r  -  x_i (sum_j x_j g_j dy_j) / (d r^3),

    which is the Jacobian-vector product written out rather than the Jacobian
    built: the second term is a rank-one correction and costs O(d).

    Note on eps.  D-5.1 step 7 reads the second term as the projection that
    annihilates the radial component, and it does so exactly when eps = 0,
    where d r^2 is exactly x.x.  Carrying eps in the denominator, as the
    forward does, leaves x.dx = eps (x.(g*dy)) / r^3, which is O(eps) and not
    zero.  The gradient here matches the forward that was actually computed,
    which is the property a gradient check tests.
    """
    x = np.asarray(x, dtype=float)
    g = np.asarray(g, dtype=float)
    dy = np.asarray(dy, dtype=float)
    d = x.shape[-1]

    r = np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    xhat = x / r

    gdy = g * dy
    radial = (x * gdy).sum(axis=-1, keepdims=True)
    dx = gdy / r - x * radial / (d * r ** 3)

    dg = dy * xhat
    if dg.ndim > 1:
        dg = dg.reshape(-1, d).sum(axis=0)
    return dx, dg
