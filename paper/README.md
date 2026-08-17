# Paper Workspace

`paper/` is now source-first. Keep LaTeX source, bibliography/class files, build helpers, docs, and the publication figure surface here.

Canonical locations:

- `paper/main.tex`: paper entrypoint
- `paper/scripts/`: canonical Python and shell scripts
- `paper/docs/`: notes and checklists
- `paper/build/`: LaTeX build artifacts
- `results/paper/analyses/`: generated CSV/JSON/JSONL/TXT outputs
- `results/paper/figures/`: source-of-truth generated figure assets
- `results/paper/synthetic_traces/`: generated synthetic trace outputs
- `artifacts/paper/models/`: trained model bundles

Compatibility:

- Old script paths under `paper/` and `paper/figures/` are kept as temporary symlinks during the migration.
- `paper/figures/` is the publication surface used by LaTeX. Use `make publish-figures` to copy referenced figure files from `results/paper/figures/`.

Common commands:

```sh
make -C paper check-paths
make -C paper publish-figures
make -C paper pdf
```
