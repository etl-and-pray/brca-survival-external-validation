"""Convert percent-format notebook sources (notebooks/*.py) to .ipynb without extra dependencies.

Usage:  python scripts/make_notebooks.py
The .py files are the source of truth (clean diffs); the .ipynb files are generated.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def split_cells(text: str) -> list[tuple[str, list[str]]]:
    cells: list[tuple[str, list[str]]] = []
    kind, buf = None, []
    for line in text.splitlines():
        if line.startswith("# %%"):
            if kind is not None:
                cells.append((kind, buf))
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        elif kind is not None:
            buf.append(line)
    if kind is not None:
        cells.append((kind, buf))
    return cells


def to_source(kind: str, lines: list[str]) -> list[str]:
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    if kind == "markdown":
        lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])


def convert(path: Path) -> Path:
    cells = []
    for kind, lines in split_cells(path.read_text(encoding="utf-8")):
        src = to_source(kind, lines)
        if not src:
            continue
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = path.with_suffix(".ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    for p in sorted(NB_DIR.glob("*.py")):
        print("wrote", convert(p).relative_to(NB_DIR.parent))
