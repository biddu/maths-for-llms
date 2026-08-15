import numpy as np
from exercises.ch03.solution import causal_mha


def test_causal_mha_matches_torch():
    torch = __import__("pytest").importorskip("torch")
    rng = np.random.default_rng(6)
    s, d, h = 11, 16, 4
    X = rng.normal(size=(s, d)) * 0.4
    Wq, Wk, Wv, Wo = (rng.normal(size=(d, d)) * 0.3 for _ in range(4))
    got = causal_mha(X, Wq, Wk, Wv, Wo, h)
    t = lambda M: torch.tensor(M, dtype=torch.float64)
    q = (t(X) @ t(Wq)).view(s, h, d // h).transpose(0, 1)
    k = (t(X) @ t(Wk)).view(s, h, d // h).transpose(0, 1)
    v = (t(X) @ t(Wv)).view(s, h, d // h).transpose(0, 1)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = (o.transpose(0, 1).reshape(s, d) @ t(Wo)).numpy()
    assert np.abs(got - ref).max() < 1e-5
