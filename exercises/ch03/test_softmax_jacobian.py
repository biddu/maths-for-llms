import numpy as np
from exercises.ch03.solution import softmax_jacobian


def test_jacobian_matches_autograd():
    torch = __import__("pytest").importorskip("torch")
    rng = np.random.default_rng(5)
    z = rng.normal(size=12)
    J = softmax_jacobian(z)
    assert np.allclose(J @ np.ones(12), 0.0, atol=1e-12), "1 lies in the kernel"
    assert np.linalg.matrix_rank(J, tol=1e-10) == 11
    ref = torch.autograd.functional.jacobian(
        lambda t: torch.softmax(t, -1), torch.tensor(z)).numpy()
    assert np.abs(J - ref).max() < 1e-8
