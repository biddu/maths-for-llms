"""E-8.10.  Perplexity is not comparable across tokenizers; bits per byte is.

The two bytes-per-token figures are real: they are what GPT-2's and Llama-3's
tokenizers measure on the committed held-out sample, recorded in
figs/data/fig82_tokenizers.json.  The two models are given slightly different
true qualities, 0.7760 and 0.7707 bits per byte, so the test is not circular:
model B really is the better model, by 0.7%.

The point of the exercise is that perplexity says the opposite.
"""
import numpy as np
from exercises.ch08.solution import bits_per_byte, perplexity

LN2 = np.log(2.0)
# (name, measured bytes/token, the model's true quality in bits/byte)
A = ("GPT-2 tokenizer", 3.5284, 0.7760)
B = ("Llama-3 tokenizer", 4.0796, 0.7707)


def _ce_nats(bpb, bpt):
    return bpb * bpt * LN2


def test_tokenizer_invariance():
    ce_a, ce_b = _ce_nats(A[2], A[1]), _ce_nats(B[2], B[1])
    bpb_a, bpb_b = bits_per_byte(ce_a, A[1]), bits_per_byte(ce_b, B[1])
    ppl_a, ppl_b = perplexity(ce_a), perplexity(ce_b)

    assert abs(bpb_a - bpb_b) < 0.02, "bits per byte must agree to 0.02"
    assert abs(ppl_a / ppl_b - 1) > 0.15, "perplexity must not"


def test_perplexity_ranks_them_the_wrong_way_round():
    """The trap, stated as an assertion.  Model A reports the lower perplexity
    and is the worse model; only bits per byte gets the ordering right."""
    ce_a, ce_b = _ce_nats(A[2], A[1]), _ce_nats(B[2], B[1])
    assert perplexity(ce_a) < perplexity(ce_b)
    assert bits_per_byte(ce_a, A[1]) > bits_per_byte(ce_b, B[1])


def test_round_trip_of_the_chapter_8_box():
    """The four coordinates of Figure 8.1, at Model D's 2.03 nats."""
    ce = 2.03
    assert abs(perplexity(ce) - 7.614086) < 1e-5
    assert abs(bits_per_byte(ce, 3.8) - 0.770703) < 1e-6
    assert abs(perplexity(ce) - 2 ** (ce / LN2)) < 1e-9
