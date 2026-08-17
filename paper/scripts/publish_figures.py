#!/usr/bin/env python3
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

from paths import PAPER_DIR, PAPER_FIGURES_DIR, PAPER_RESULTS_FIGURES_DIR


INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def referenced_figure_paths() -> list[Path]:
    refs: set[Path] = set()
    for tex_path in PAPER_DIR.glob("*.tex"):
        text = tex_path.read_text(encoding="utf-8")
        for rel in INCLUDEGRAPHICS_RE.findall(text):
            if rel.startswith("figures/"):
                refs.add(Path(rel.removeprefix("figures/")))
    return sorted(refs)


def publish() -> int:
    missing: list[Path] = []
    for rel in referenced_figure_paths():
        source = PAPER_RESULTS_FIGURES_DIR / rel
        target = PAPER_FIGURES_DIR / rel
        if not source.exists():
            missing.append(source)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    if missing:
        for path in missing:
            print(f"missing source figure: {path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(publish())
