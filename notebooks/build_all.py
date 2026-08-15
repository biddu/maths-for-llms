"""Rebuild the companion notebooks.

    python3 notebooks/build_all.py            all sixteen
    python3 notebooks/build_all.py 3 11       those chapters only

Each chapter's content lives in `notebooks/nb_chNN.py`, which exposes CHAPTER,
SLUG, TITLE, BLURB and SECTIONS.  This file only assembles them, so adding a
chapter means adding one module and nothing else.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _nbbuild  # noqa: E402


def load(path: str):
    spec = importlib.util.spec_from_file_location(
        os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str]) -> int:
    wanted = {int(a) for a in argv} if argv else None
    mods = sorted(
        (f for f in os.listdir(HERE) if re.fullmatch(r"nb_ch\d\d\.py", f)),
        key=lambda f: int(f[5:7]))
    if not mods:
        sys.exit("no nb_chNN.py modules found in notebooks/")
    built = 0
    for f in mods:
        n = int(f[5:7])
        if wanted and n not in wanted:
            continue
        m = load(os.path.join(HERE, f))
        p = _nbbuild.build(m.CHAPTER, m.SLUG, m.TITLE, m.BLURB, m.SECTIONS)
        print(f"{os.path.basename(p):34s} {len(m.SECTIONS)} sections")
        built += 1
    print(f"\n{built} notebooks written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
