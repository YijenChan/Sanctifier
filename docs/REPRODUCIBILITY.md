# Reproducibility boundary

## Reported environment

- Python 3.12.9
- NVIDIA RTX 3090
- PyTorch 2.5.1+cu124
- Transformers 4.57.6 for Qwen2.5
- Transformers 4.53.2-compatible implementation for Qwen2-Audio
- Sentence Transformers 3.4.1
- PyArrow 25.0.1
- FP16 inference and greedy decoding

Model revisions and dataset snapshots should be recorded in the local run manifest because their licenses may prevent redistribution. The repository intentionally contains no checkpoint, dataset, ASR cache, label file, prediction file, or development trace.

## Label isolation

The `score-prior`, `prepare`, `verify`, and `reconcile` stages do not open `data.labels`. Only `evaluate` reads that path. For official evaluation, keep the label file outside all inference directories and run `evaluate` only after every decision file has been frozen.

## Expected private inputs

- timestamped Whisper-large-v3 transcripts;
- label-free inference index with sample IDs, durations, questions, and audio hashes;
- greedy full-transcript and retrieved-view outputs from `run_aligned_reader.py`;
- locally licensed Qwen2.5 and Qwen2-Audio checkpoints;
- a scorer-only JSONL file containing `sample_id` and `ground_truth`.

Every `<EDIT_ME>` path in `configs/sanctifier.example.yaml` must be replaced. The example configuration contains no machine-specific path.

## Pre-release checks

```powershell
rg -n "API_KEY|BEGIN .*PRIVATE KEY|sk-[A-Za-z0-9_-]+|ghp_[A-Za-z0-9]+" .
rg -n "[A-Z]:\\\\|F:/ICASSP" .
python -m unittest discover -s tests -v
git status --short
```

The first two searches should report only the search examples in this document or ignore rules, never a credential or private path. See `METHOD_TO_CODE.md` for the consolidation regression check and `EVALUATION_PROTOCOL.md` for the frozen test boundary.
