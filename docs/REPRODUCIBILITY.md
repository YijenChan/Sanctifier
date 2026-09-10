# Reproducibility and evaluation boundary

The frozen protocol is in `EVALUATION_PROTOCOL.md`. Development decisions and thresholds must be fixed before official test scoring. The final scorer is the only component permitted to read ground-truth labels. Selection, triggering, audio verification, reconciliation, and fallback behavior must remain label-blind.

The original environment used Python 3.12.9 and an NVIDIA RTX 3090. Before publication, export exact installed package versions and model revisions; `requirements.txt` currently records dependencies, not a fully pinned environment.

Expected private inputs include timestamped ASR caches, transcript-view predictions, locally licensed Qwen model directories, and scorer labels. These are omitted from this package. Example YAML files identify every required path.

Recommended pre-release checks:

```powershell
rg -n "API_KEY|BEGIN .*PRIVATE KEY|sk-[A-Za-z0-9_-]+|F:\\ICASSP|[A-Z]:\\" .
git status --short
pytest -q
```

Experimental outputs are not distributed in this repository. When reconstructing the analysis, retain every registered comparison, including failed and non-improving controls, rather than selecting favorable runs.
