#!/usr/bin/env python3
"""Build deterministic, audio-disjoint tune/holdout views from one train shard."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def audio_digest(array: np.ndarray, rate: int) -> str:
    digest = hashlib.sha256()
    digest.update(str(rate).encode("ascii"))
    digest.update(np.ascontiguousarray(array, dtype=np.float32).view(np.uint8))
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--duration", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tune", type=int, default=100)
    parser.add_argument("--holdout", type=int, default=100)
    parser.add_argument("--seed", type=int, default=25)
    args = parser.parse_args()

    table = pq.read_table(args.parquet, columns=["audio_array", "sampling_rate", "question", "label"])
    groups: dict[str, list[int]] = {}
    rates: dict[int, int] = {}
    question_hashes: dict[int, str] = {}
    labels: dict[int, str] = {}
    for index, row in enumerate(table.to_pylist()):
        audio = np.asarray(row["audio_array"], dtype=np.float32)
        rate = int(row["sampling_rate"])
        digest = audio_digest(audio, rate)
        groups.setdefault(digest, []).append(index)
        rates[index] = rate
        question_hashes[index] = hashlib.sha256(str(row["question"]).encode("utf-8")).hexdigest()
        labels[index] = str(row["label"])

    rng = np.random.default_rng(args.seed)
    group_ids = sorted(groups)
    rng.shuffle(group_ids)

    def take_groups(start: int, target: int) -> tuple[list[int], int]:
        selected: list[int] = []
        cursor = start
        while cursor < len(group_ids) and len(selected) < target:
            candidate = groups[group_ids[cursor]]
            if len(selected) + len(candidate) <= target:
                selected.extend(candidate)
            cursor += 1
        if len(selected) != target:
            raise RuntimeError(f"Could not form exactly {target} samples from whole audio groups; got {len(selected)}")
        return sorted(selected), cursor

    tune, cursor = take_groups(0, args.tune)
    holdout, _ = take_groups(cursor, args.holdout)
    parquet_name = args.parquet.name

    def inference(indices: list[int]) -> list[dict]:
        return [{
            "sample_id": f"{args.duration}/{parquet_name}:{index:04d}",
            "row_index": index,
            "sampling_rate": rates[index],
            "question_sha256": question_hashes[index],
            "audio_group_fingerprint": next(key for key, values in groups.items() if index in values),
        } for index in indices]

    def scorer(indices: list[int]) -> list[dict]:
        return [{
            "sample_id": f"{args.duration}/{parquet_name}:{index:04d}",
            "ground_truth": labels[index],
        } for index in indices]

    write_jsonl(args.output_dir / "tune_inference.jsonl", inference(tune))
    write_jsonl(args.output_dir / "tune_scorer_labels.jsonl", scorer(tune))
    write_jsonl(args.output_dir / "holdout_inference.jsonl", inference(holdout))
    write_jsonl(args.output_dir / "holdout_scorer_labels.jsonl", scorer(holdout))
    summary = {
        "duration_minutes": args.duration,
        "seed": args.seed,
        "tune_samples": len(tune),
        "holdout_samples": len(holdout),
        "tune_audio_groups": len({row["audio_group_fingerprint"] for row in inference(tune)}),
        "holdout_audio_groups": len({row["audio_group_fingerprint"] for row in inference(holdout)}),
        "shared_audio_groups": len(
            {row["audio_group_fingerprint"] for row in inference(tune)}
            & {row["audio_group_fingerprint"] for row in inference(holdout)}
        ),
        "labels_in_inference": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
