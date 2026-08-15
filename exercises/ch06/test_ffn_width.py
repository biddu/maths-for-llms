import pytest
from exercises.ch06.solution import intermediate_size


def test_llama3_widths():
    """E-6.9.  The two published widths, from the same four lines of code."""
    assert intermediate_size(4096) == 14336
    assert intermediate_size(8192) == 28672


def test_the_pipeline_is_not_commutative():
    """Rounding up to a multiple of 1024 before applying the multiplier gives a
    different answer, which is why the box prints the steps in order."""
    assert intermediate_size(4096, multiplier=1.0, multiple_of=1) == 10922
    assert intermediate_size(4096, multiplier=1.0, multiple_of=1024) == 11264
    assert intermediate_size(4096) != int(1.3 * 11264)


@pytest.mark.parametrize("d,expected", [(1024, 4096), (2048, 7168), (4096, 14336), (8192, 28672)])
def test_family(d, expected):
    assert intermediate_size(d) == expected
