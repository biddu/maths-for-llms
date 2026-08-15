import numpy as np
import pytest


@pytest.mark.slow
def test_yarn_beats_pi():
    pytest.importorskip("torch")
    pytest.skip("E-4.13: train the 512-token toy model, then remove this skip")
