"""Shared machinery for building the sixteen companion notebooks.

Every notebook in this directory is GENERATED, from `notebooks/nb_chNN.py`, and
the generator is committed alongside the output.  That is deliberate and it is
the same contract the figures keep: a notebook nobody can rebuild is a liability
in a second edition, and a notebook edited by hand drifts from the chapter it
claims to reproduce.

    python3 notebooks/build_all.py          rebuild all sixteen
    python3 notebooks/build_all.py 3 11     rebuild those chapters only

Three rules every notebook obeys, and CI enforces all three by executing them.

  1. **Every section asserts.**  A cell that prints and asserts nothing is not
     doing its job.  If a chapter claims a number, the notebook recomputes it
     and fails when it moves.
  2. **NumPy and SciPy only.**  Nothing here imports torch or transformers and
     nothing downloads a checkpoint.  Where a measurement is needed it comes
     from `figs/data/`, which is committed.
  3. **Numbers come from `arith/`, never from the page.**  A notebook that
     hard-codes 6,979,588,096 agrees with the book until someone edits the
     model table.  One that calls `non_embedding(MODEL_D)` cannot.
"""
from __future__ import annotations

import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Prepended to every notebook.  The sys.path line is what lets `import arith`
# work from a fresh clone with no install step, which is the point.
PREAMBLE = """\
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.insert(0, os.path.abspath(os.getcwd()))
np.set_printoptions(precision=6, suppress=True)
print("numpy", np.__version__)"""


def header(chapter: int, title: str, blurb: str) -> str:
    return (
        f"# Chapter {chapter} — {title}\n\n"
        f"{blurb}\n\n"
        "Every section below recomputes a result the chapter states, and asserts it. "
        "Nothing is transcribed from the page: the numbers come from the `arith/` "
        "modules and from the measurements committed under `figs/data/`, so this "
        "notebook and the book cannot drift apart without this notebook failing.\n\n"
        "Run it top to bottom. NumPy and SciPy only; no checkpoint is downloaded."
    )


def build(chapter: int, slug: str, title: str, blurb: str, sections) -> str:
    """sections: iterable of (number, heading, markdown, code)."""
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(header(chapter, title, blurb)))
    nb.cells.append(nbf.v4.new_code_cell(PREAMBLE))
    for num, heading, md, code in sections:
        nb.cells.append(nbf.v4.new_markdown_cell(f"## §{num} — {heading}\n\n{md}"))
        nb.cells.append(nbf.v4.new_code_cell(code.rstrip()))
    nb.cells.append(nbf.v4.new_markdown_cell(
        "---\n\nEvery assertion above passed, so every number this chapter prints "
        "still follows from the code that produced it."))
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.12"}
    path = os.path.join(HERE, f"ch{chapter:02d}_{slug}.ipynb")
    with open(path, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    return path
