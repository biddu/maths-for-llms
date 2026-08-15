"""Your solutions for Chapter 4's [C] exercises."""
import numpy as np


def apply_rope(x, m, theta):
    """E-4.10.  Rotate each 2-D pair of x by m*theta_i.

    Pairs are adjacent coordinates, (x_0, x_1), (x_2, x_3), and so on, which is
    the convention `rope_complex` needs and the one the wavelength ladder of
    section 4.4 indexes.  The angles are formed in float64 even when x is
    float32: m*theta_0 reaches 8192 radians at the top of the trained context,
    and rounding the angle rather than the coordinate is what costs precision.
    """
    x = np.asarray(x)
    theta = np.asarray(theta, dtype=float)
    ang = m * theta
    c, s = np.cos(ang), np.sin(ang)
    even, odd = x[..., 0::2], x[..., 1::2]
    out = np.empty(x.shape, dtype=np.result_type(x.dtype, float))
    out[..., 0::2] = even * c - odd * s
    out[..., 1::2] = even * s + odd * c
    return out


def rope_complex(x, m, theta):
    """E-4.11.  The same rotation as a complex multiply.

    A rotation of the plane by an angle is multiplication by e^{i angle}, so
    reading each pair as one complex number turns the block-diagonal R_m into
    a single elementwise product.
    """
    x = np.asarray(x)
    theta = np.asarray(theta, dtype=float)
    z = x[..., 0::2] + 1j * x[..., 1::2]
    y = z * np.exp(1j * m * theta)
    out = np.empty(x.shape, dtype=np.result_type(x.dtype, float))
    out[..., 0::2] = y.real
    out[..., 1::2] = y.imag
    return out


def yarn_scaled_theta(theta, s, L, alpha_y=1.0, beta_y=32.0):
    """E-4.12.  YaRN's per-band interpolation. Return (theta_prime, gamma).

    Band i has wavelength lambda_i = 2 pi / theta_i, so it completes
    r_i = L / lambda_i turns over the trained context L.  YaRN interpolates a
    band in proportion to how little it rotated: gamma = 1 above beta_y turns
    (extrapolate, leave theta alone), gamma = 0 below alpha_y turns (full
    position interpolation, theta / s), linear in between.  L is the *trained*
    context; passing the extended one sends every gamma to 1 and the whole
    method becomes a silent no-op.
    """
    theta = np.asarray(theta, dtype=float)
    turns = L * theta / (2 * np.pi)                       # r_i = L / lambda_i
    gamma = np.clip((turns - alpha_y) / (beta_y - alpha_y), 0.0, 1.0)
    theta_prime = theta * (gamma + (1.0 - gamma) / s)
    return theta_prime, gamma
