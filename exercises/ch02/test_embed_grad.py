import numpy as np
from exercises.ch02.solution import embedding_backward


def test_scatter_add_matches_autograd():
    torch = __import__("pytest").importorskip("torch")
    rng = np.random.default_rng(3)
    V, d = 50, 8
    ids = np.array([3, 7, 3, 41, 7, 3])
    dX = rng.normal(size=(ids.size, d))
    g = embedding_backward(ids, dX, V)
    assert g.shape == (V, d)
    assert np.count_nonzero(np.abs(g).sum(1)) == len(set(ids.tolist()))
    emb = torch.nn.Embedding(V, d)
    out = emb(torch.tensor(ids))
    out.backward(torch.tensor(dX, dtype=out.dtype))
    assert np.allclose(g, emb.weight.grad.numpy(), atol=1e-6)
