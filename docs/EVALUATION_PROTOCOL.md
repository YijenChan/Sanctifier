# Frozen evaluation protocol

The official evaluation contains 750 questions for each of the 2-, 5-, and 10-minute YODAS2-MCQA conditions. All controlled methods must use the same frozen timestamped Whisper cache, sample index, questions, and text-reader checkpoint.

Two reader interfaces are intentionally distinguished. The earlier free-generation ASR-Full reference produces 643/750, 619/750, and 595/750 correct answers. Sanctifier's controlled transcript prior uses normalized next-token scores over the four choice digits and produces 646/750, 631/750, and 604/750. The proposed acoustic correction must be compared with the latter because it is the identical prior used inside the final system. Do not attribute the reader-interface difference to raw-audio verification.

Development choices—including prompts, thresholds, clip length, random seeds, parsing, and fallback rules—must be fixed without inspecting official test labels. The official labels are isolated from all preparation, evidence selection, triggering, verification, and reconciliation stages and are read only by the final scorer.

The verification budget is enforced per example. Overlapping waveform intervals are merged before duration is counted. Report both the configured cap and actual deduplicated speech-access ratio. Full-recording ASR accesses the complete waveform and is not included in the verification-only ratio.

The primary comparison reports exact-choice accuracy by duration and jointly, paired wrong-to-right/right-to-wrong transitions against the frozen ASR prior, and paired confidence intervals. Any implementation error found after scoring must be logged and all affected samples rerun under one corrected version; test-set observations must not be used for retuning.
