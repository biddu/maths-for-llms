import pytest


@pytest.mark.slow
def test_postnorm_needs_warmup():
    pytest.importorskip("torch")
    pytest.skip("E-5.12: train the six-layer toy model, then remove this skip")
