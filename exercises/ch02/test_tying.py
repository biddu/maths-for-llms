from exercises.ch02.solution import param_count


def test_tied_and_untied_totals():
    assert param_count(V=128256, d=4096, L=32, d_ff=14336, tied=False) == 8_030_261_248
    assert param_count(V=128256, d=2048, L=16, d_ff=8192, tied=False) == 1_498_482_688
    assert param_count(V=128256, d=2048, L=16, d_ff=8192, tied=True) == 1_235_814_400
