from __future__ import annotations

from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent

PAPER_BUILD_DIR = PAPER_DIR / "build"
PAPER_DOCS_DIR = PAPER_DIR / "docs"
PAPER_SCRIPTS_DIR = PAPER_DIR / "scripts"
PAPER_FIGURES_DIR = PAPER_DIR / "figures"

PAPER_RESULTS_DIR = REPO_ROOT / "results" / "paper"
PAPER_RESULTS_ANALYSES_DIR = PAPER_RESULTS_DIR / "analyses"
PAPER_RESULTS_FIGURES_DIR = PAPER_RESULTS_DIR / "figures"
PAPER_RESULTS_SYNTHETIC_TRACES_DIR = PAPER_RESULTS_DIR / "synthetic_traces"
PAPER_RESULTS_INVENTORY_DIR = PAPER_RESULTS_DIR / "inventory"

PAPER_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "paper"
PAPER_MODELS_DIR = PAPER_ARTIFACTS_DIR / "models"


def ensure_paper_layout() -> None:
    for path in (
        PAPER_BUILD_DIR,
        PAPER_DOCS_DIR,
        PAPER_SCRIPTS_DIR,
        PAPER_FIGURES_DIR,
        PAPER_RESULTS_DIR,
        PAPER_RESULTS_ANALYSES_DIR,
        PAPER_RESULTS_FIGURES_DIR,
        PAPER_RESULTS_SYNTHETIC_TRACES_DIR,
        PAPER_RESULTS_INVENTORY_DIR,
        PAPER_ARTIFACTS_DIR,
        PAPER_MODELS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def paper_model_dir(name: str) -> Path:
    return PAPER_MODELS_DIR / name


def analysis_prefix(stem: str) -> Path:
    return PAPER_RESULTS_ANALYSES_DIR / stem


def figures_results_path(*parts: str) -> Path:
    return PAPER_RESULTS_FIGURES_DIR.joinpath(*parts)


def figures_publish_path(*parts: str) -> Path:
    return PAPER_FIGURES_DIR.joinpath(*parts)


def synthetic_traces_path(*parts: str) -> Path:
    return PAPER_RESULTS_SYNTHETIC_TRACES_DIR.joinpath(*parts)
