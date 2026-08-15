"""Your solutions for Chapter 5's [C] exercises."""


def rho_per_layer(activations):
    """E-5.10.  |mean(x)|/RMS(x) over the feature axis, per layer."""
    raise NotImplementedError


def rmsnorm_forward(x, g, eps=1e-5):
    """E-5.11.  RMSNorm forward, equation (5.4)."""
    raise NotImplementedError


def rmsnorm_backward(x, g, dy, eps=1e-5):
    """E-5.11.  The Jacobian of D-5.1, applied to dy. Return (dx, dg)."""
    raise NotImplementedError
