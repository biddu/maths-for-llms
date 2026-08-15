"""Your solutions for this chapter's [C] exercises.

Every function below raises NotImplementedError, so every test in this directory
fails on a fresh clone.  Making them pass is the exercise.  Worked solutions are
on the `solutions` branch; Appendix C prints the answer and the path, not the
code.
"""


def embedding_backward(ids, dX, V):
    """E-2.9.  Scatter-add the upstream gradient into a V x d matrix."""
    raise NotImplementedError


def centred_cosine(W):
    """E-2.10.  Cosine similarity after subtracting the row mean of W."""
    raise NotImplementedError


def all_but_the_top(W, k=1):
    """E-2.10 (extended).  Cosine after removing the mean and the top k
    principal directions.  Centring alone is necessary and not sufficient; see
    measure/README.md."""
    raise NotImplementedError


def param_count(V, d, L, d_ff, tied):
    """E-2.11.  Reproduce both figures in the Chapter 2 arithmetic box."""
    raise NotImplementedError
