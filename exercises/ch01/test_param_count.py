from exercises.ch01.solution import count_params


def test_model_d_total():
    assert count_params(L=32, d=4096, h=32, d_h=128, n_kv=8,
                        d_ff=14336, V=128256, tied=False) == 8_030_261_248
    assert count_params(L=32, d=4096, h=32, d_h=128, n_kv=8,
                        d_ff=14336, V=128256, tied=True) == 7_504_924_672
