# Frozen evaluation protocol

## Primary data boundary

YODAS2-MCQA contains 750 questions for each 2-, 5-, and 10-minute condition. Controlled comparisons use the same timestamped Whisper cache, sample index, questions, and Qwen2.5 checkpoint. Development choices, including prompts, parsing, thresholds, clip cap, random seeds, and fallback rules, must be fixed before official test scoring.

The official labels are isolated from all preparation, retrieval, triggering, acoustic verification, and reconciliation. In the canonical release, only:

```bash
python src/sanctifier.py evaluate --config <CONFIG>
```

opens `data.labels`.

## Reader interfaces

Two reader interfaces are intentionally distinguished:

- the greedy full/local hypotheses produced by `tools/run_aligned_reader.py` are used only by the hard cross-view trigger;
- the final transcript prior is produced by normalized next-token scores over the four choice digits in `sanctifier.py score-prior`.

The controlled transcript prior obtains 646/750, 631/750, and 604/750 correct answers. Sanctifier must be compared with this identical prior. The earlier free-generation ASR-Full reference obtains 643/750, 619/750, and 595/750; its interface difference must not be attributed to raw-audio verification.

## Frozen paper configuration

```yaml
trigger_policy: cross_view
sparse_words_per_second: 0.1
verification_clip_seconds: 20.0
verification_sar_cap: 0.20
selection_policy: query
verifier_choices_visible: false
require_attribution: true
```

The hard trigger is `global_greedy != local_greedy OR words_per_second < 0.1`. One clip is centered on the highest-scoring retrieved window, with duration `min(20 seconds, 0.20 * recording_duration)`. The verifier receives the question stem without answer choices. A changed choice is accepted only when its nonempty supporting phrase occurs in the verifier record after lowercase alphanumeric normalization and the record is not marked irrelevant or uncertain.

## Accounting and reporting

The verification budget is enforced per example. Report the configured clip/SAR caps and the observed SAR distribution, assigning zero to untriggered examples. Verification SAR excludes the complete-recording ASR pass and therefore is not an end-to-end audio-access ratio.

The primary comparison reports exact-choice accuracy by duration and jointly, paired W2R/R2W transitions against the frozen prior, and paired audio-cluster bootstrap intervals (10,000 resamples, seed 25). Random-trigger controls use seeds 11, 25, and 47 and match the frozen per-duration trigger counts.

If an implementation error is found after scoring, log it and rerun every affected sample under one corrected version. Test observations must never be used to retune the system.
