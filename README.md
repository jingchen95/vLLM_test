# vLLM_test

Research repository for energy-efficient heterogeneous vLLM serving, including
hardware characterization, XpYd prefill/decode validation, scheduler simulation,
paper analyses, and reproducibility artifacts.

## Repository map

- `paper/`: paper source, figures, experiment drivers, scheduler/runtime code,
  and phase-specific runbooks.
- `paper/scripts/`: the main maintained Python implementation.
- `paper/configs/`: checked-in experiment configurations.
- `artifacts/paper/models/`: canonical committed predictor bundles and provenance.
- `results/`: compact committed analyses plus ignored local experiment outputs.
- `tests/`: CPU-only regression and validation tests.
- `docs/`: repository audits, research-gap analysis, handoff notes, and historical
  planning documents.
- `Phase2_Results_*/master_results.csv`: compact canonical hardware tables; their
  ignored raw monitor corpora should be archived externally.

The Python and shell files at the repository root are stable command-line entry
points used by existing runbooks and launchers. They remain at the root to avoid
breaking reproducibility commands; new implementation code belongs under
`paper/scripts/`.

## Documentation

- [Research gap analysis](docs/RESEARCH_GAP_ANALYSIS.md)
- [Codebase and model-provenance audit](docs/CODEBASE_AUDIT.md)
- [Current handoff](docs/HANDOFF_FOR_CHATGPT.md)
- [GitHub curation plan](docs/GITHUB_UPLOAD_PLAN.md)
- [Related-work comparison](docs/RelatedWorkComparison_updated_2026-08-17.md)
- [Paper and experiment documentation](paper/README.md)
- [XpYd runtime design](paper/XPYD_RUNTIME_DESIGN.md)
- [Phase 3C substrate validation](paper/XPYD_PHASE3C_SUBSTRATE.md)

Large raw or regenerable local artifacts are intentionally excluded from Git.
See [the external-artifact manifest](external_artifacts_manifest.tsv) before
deleting or archiving hardware measurements.
