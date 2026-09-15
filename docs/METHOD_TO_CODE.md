# Method-to-code audit

This ledger identifies the code path corresponding to the submitted methodology. It is intended to prevent development candidates from being mistaken for the final algorithm.

## Canonical path

| Manuscript statement | Implementation | Configuration |
|---|---|---|
| Whisper produces a full timestamped transcript | precomputed ASR cache consumed by `sanctifier.py` | `data.cache_dir`, `data.transcripts_dir` |
| The transcript prior uses normalized next-token scores over four choice digits | `stage_score_prior`, `prior_prompt`, `score_choices` | `models.text_llm` |
| The stability hypotheses use greedy decoding on full and top-three retrieved transcript views | `tools/run_aligned_reader.py` | `--variant full` and `--variant rag` |
| A disagreement or transcript density below 0.1 words/s triggers verification | `trigger_decision`, `stage_prepare` | `trigger_policy: cross_view`, `sparse_words_per_second: 0.1` |
| One clip is centered on the highest-scoring retrieved window | `bounded_interval`, `stage_prepare` | `selection_policy: query` |
| Clip duration is `min(L, beta*T)` | `bounded_interval` | `verification_clip_seconds: 20.0`, `verification_sar_cap: 0.20` |
| The verifier sees the question stem without choices | `question_stem`, `stage_verify` | `verifier_choices_visible: false` |
| Irrelevant or uncertain records preserve the prior | `note_is_relevant`, `stage_reconcile` | fixed parser |
| A changed answer needs a normalized verbatim substring from the acoustic record | `quote_is_attributed`, `should_accept_update` | `require_attribution: true` |
| Labels are unavailable to selection, verification, and reconciliation | labels are opened only by `stage_evaluate` | `data.labels` |
| Reported uncertainty uses a paired audio-cluster bootstrap | `paired_cluster_interval`, `stage_evaluate` | 10,000 resamples, seed 25 |

## Consolidation check

The reported experiments were originally executed through three development files:

1. the hard trigger, interval selection, and choice-blind verifier from `aligned_pipeline_v1.py`;
2. the normalized full-transcript prior from `aligned_pipeline_v2.py`; and
3. the evidence-attributed reconciliation from `aligned_pipeline_v3.py`.

The release consolidates only those executed stages into `src/sanctifier.py`. Uncertainty thresholds, Jensen-Shannon scoring, acoustic score fusion, and option-token support were development candidates and are not part of the submitted method.

The consolidation was regression-checked on the frozen artifacts used for the paper:

- trigger decisions, selected intervals, and SAR matched all 2,250 YODAS2-MCQA examples;
- parsed candidates, evidence quotations, and accept/preserve decisions matched all 2,250 examples;
- the model prompts and deterministic decoding settings in the executed stages were preserved.

These checks establish behavioral equivalence for the label-free artifacts available from the executed runs. They do not replace an independent end-to-end rerun from model checkpoints.

## Interpretation boundary

The attribution check establishes provenance to the verifier's text record, not truthfulness of that record. Verification SAR measures the selected waveform sent to the audio-language model and excludes the full-recording ASR pass. Both boundaries are stated in the manuscript and retained in the code comments and evaluation output.
