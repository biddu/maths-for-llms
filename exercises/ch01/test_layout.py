import numpy as np
from exercises.ch01.solution import linear_backward


def test_linear_grads_match_autograd():
    torch = __import__("pytest").importorskip("torch")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(7, 5)); W = rng.normal(size=(5, 3)); dY = rng.normal(size=(7, 3))
    dW, dX = linear_backward(X, W, dY)
    assert dW.shape == W.shape and dX.shape == X.shape, "denominator layout"
    tX = torch.tensor(X, requires_grad=True); tW = torch.tensor(W, requires_grad=True)
    (tX @ tW).backward(torch.tensor(dY))
    assert np.allclose(dW, tW.grad.numpy(), atol=1e-10)
    assert np.allclose(dX, tX.grad.numpy(), atol=1e-10)
