# Sanctifier

Reference implementation of **Sanctifier: Budgeted Raw-Audio Verification for Long-Form Spoken Question Answering**.

Sanctifier first forms a full-transcript prior, then compares greedy answers from the full transcript and a question-relevant local transcript view. A disagreement, or an unusually sparse transcript, triggers one duration-capped waveform audit. The audio-language model sees the question stem but not the choices. A frozen reconciliation reader may revise the prior only when it returns a supporting phrase attributable to the acoustic record.

![Sanctifier framework](assets/framework.png)

## Paper-to-code correspondence

The release has one canonical entry point, `src/sanctifier.py`:

| Paper component | Command stage | Output |
|---|---|---|
| Full-transcript prior, Eq. (1) | `score-prior` | `priors.jsonl` |
| Cross-view trigger and bounded clip, Eqs. (3)-(4) | `prepare` | `prepared.jsonl` |
| Choice-blind acoustic record, Eq. (5) | `verify` | `verifications.jsonl` |
| Evidence-attributed update, Eqs. (6)-(7) | `reconcile` | `decisions.jsonl` |
| Label-isolated metrics and paired cluster interval | `evaluate` | `evaluation.json` |

`tools/run_aligned_reader.py` creates the two greedy transcript-view hypotheses and the top-three BGE windows used by `prepare`. The transcript prior is deliberately scored separately with normalized next-token scores, matching the distinction made in the paper.

The earlier public filenames `aligned_pipeline_v1.py`, `aligned_pipeline_v2.py`, and `aligned_pipeline_v3.py` reflected development lineage and contained unused candidate mechanisms. They were consolidated into the canonical entry point without changing the executed paper path. See [`docs/METHOD_TO_CODE.md`](docs/METHOD_TO_CODE.md) and the machine-readable [`paper-code-alignment.yaml`](docs/paper-code-alignment.yaml) for the equivalence audit.

## Repository scope

The repository contains system code, path-free example configuration, tests, the method-to-code ledger, and evaluation protocol. It excludes datasets, model weights, baseline repositories, ASR caches, labels, predictions, development traces, and intermediate experimental outputs.

## Structure

- `src/sanctifier.py`: canonical paper pipeline.
- `src/core.py`: atomic JSON I/O and question-stem extraction.
- `tools/run_aligned_reader.py`: frozen-prompt global/local transcript readers and BGE retrieval.
- `tools/run_coraal_rq2.py`: CORAAL-QA localization diagnostic.
- `tools/build_duration_dev_split.py`: deterministic audio-disjoint development split builder.
- `tools/score_aligned_predictions.py`: standalone label-reading scorer for transcript baselines.
- `tools/summarize_yodas.py`: pooled duration metrics and paired audio-cluster interval.
- `configs/sanctifier.example.yaml`: paper configuration with editable paths.
- `docs/EVALUATION_PROTOCOL.md`: frozen evaluation and label-isolation rules.
- `docs/REPRODUCIBILITY.md`: environment and artifact boundaries.
- `tests/test_core.py`: label-free invariants for the trigger, budget, and update contract.

## Installation

The reported runs used Python 3.12.9, PyTorch 2.5.1+cu124, Transformers 4.57.6 for the text reader, Sentence Transformers 3.4.1, and a Transformers 4.53.2-compatible Qwen2-Audio implementation. Install a CUDA-compatible PyTorch build first, then:

```bash
python -m pip install -r requirements.txt
```

Dataset and model access must follow their respective licenses. No data, checkpoint, or API credential is included.

## Reproduction outline

Copy `configs/sanctifier.example.yaml`, replace every `<EDIT_ME>` path, and keep the scorer labels outside the inference output directory.

```bash
export PYTHONPATH="$PWD/src"

# 1. Produce the frozen greedy hypotheses and BGE windows used by the trigger.
python tools/run_aligned_reader.py --cache-dir <ASR_CACHE> --index-path <INFERENCE_INDEX> --reader-path <TEXT_MODEL> --variant full --output-dir <FULL_OUTPUT>
python tools/run_aligned_reader.py --cache-dir <ASR_CACHE> --index-path <INFERENCE_INDEX> --reader-path <TEXT_MODEL> --bge-path <BGE_MODEL> --variant rag --output-dir <RAG_OUTPUT>

# 2. Run the canonical paper path. Only the last command reads labels.
python src/sanctifier.py score-prior --config configs/sanctifier.yaml
python src/sanctifier.py prepare --config configs/sanctifier.yaml
python src/sanctifier.py verify --config configs/sanctifier.yaml
python src/sanctifier.py reconcile --config configs/sanctifier.yaml
python src/sanctifier.py evaluate --config configs/sanctifier.yaml

# Pool the three duration-specific scored outputs used in the paper table.
python tools/summarize_yodas.py --scored <2MIN_SCORED> <5MIN_SCORED> <10MIN_SCORED> --output <SUMMARY_JSON>

python -m unittest discover -s tests -v
```

The registered ablations use the same code path by changing only the documented method fields: `trigger_policy`, `selection_policy`, `verification_clip_seconds`, `verifier_choices_visible`, and `require_attribution`.

## Accounting boundary

Verification SAR is the deduplicated waveform duration sent to the acoustic verifier divided by recording duration, with zero assigned to untriggered examples. It excludes the full-recording ASR pass, so it is not an end-to-end audio-access ratio.

## License and citation

No source-code license has been selected yet. Add a license and the final bibliographic entry before treating this repository as a formal software release.
