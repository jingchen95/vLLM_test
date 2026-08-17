#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

from paths import (
    PAPER_BUILD_DIR,
    PAPER_DIR,
    PAPER_DOCS_DIR,
    PAPER_FIGURES_DIR,
    PAPER_RESULTS_ANALYSES_DIR,
    PAPER_RESULTS_FIGURES_DIR,
    PAPER_RESULTS_SYNTHETIC_TRACES_DIR,
    PAPER_SCRIPTS_DIR,
    ensure_paper_layout,
)


INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def referenced_figure_paths() -> list[Path]:
    refs: set[Path] = set()
    for tex_path in PAPER_DIR.glob("*.tex"):
        text = tex_path.read_text(encoding="utf-8")
        for rel in INCLUDEGRAPHICS_RE.findall(text):
            if rel.startswith("figures/"):
                refs.add(Path(rel.removeprefix("figures/")))
    return sorted(refs)


def main() -> int:
    ensure_paper_layout()
    required_dirs = [
        PAPER_BUILD_DIR,
        PAPER_DOCS_DIR,
        PAPER_SCRIPTS_DIR,
        PAPER_FIGURES_DIR,
        PAPER_RESULTS_ANALYSES_DIR,
        PAPER_RESULTS_FIGURES_DIR,
        PAPER_RESULTS_SYNTHETIC_TRACES_DIR,
    ]
    missing_dirs = [path for path in required_dirs if not path.exists()]
    if missing_dirs:
        for path in missing_dirs:
            print(f"missing directory: {path}", file=sys.stderr)
        return 1

    missing_results = []
    missing_published = []
    for rel in referenced_figure_paths():
        if not (PAPER_RESULTS_FIGURES_DIR / rel).exists():
            missing_results.append(PAPER_RESULTS_FIGURES_DIR / rel)
        if not (PAPER_FIGURES_DIR / rel).exists():
            missing_published.append(PAPER_FIGURES_DIR / rel)

    for path in missing_results:
        print(f"missing results figure: {path}", file=sys.stderr)
    for path in missing_published:
        print(f"missing published figure: {path}", file=sys.stderr)

    return 1 if missing_results or missing_published else 0


if __name__ == "__main__":
    raise SystemExit(main())
