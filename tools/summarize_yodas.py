#!/usr/bin/env python3
"""Pool label-isolated per-duration outputs into the paper's YODAS summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core import read_jsonl, write_json_atomic
from sanctifier import paired_cluster_interval


def metrics(rows: list[dict], bootstrap_resamples: int, bootstrap_seed: int) -> dict:
    prior_correct = sum(row["prior_correct"] for row in rows)
    final_correct = sum(row["final_correct"] for row in rows)
    wrong_to_right = sum(
        (not row["prior_correct"]) and row["final_correct"] for row in rows
    )
    right_to_wrong = sum(
        row["prior_correct"] and (not row["final_correct"]) for row in rows
    )
    return {
        "samples": len(rows),
        "prior_accuracy_percent": 100.0 * prior_correct / len(rows),
        "accuracy_percent": 100.0 * final_correct / len(rows),
        "gain_percentage_points": 100.0 * (final_correct - prior_correct) / len(rows),
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "changed": sum(row["prior"] != row["final"] for row in rows),
        "triggered": sum(row["triggered"] for row in rows),
        "accepted": sum(row["accepted"] for row in rows),
        "mean_verification_sar_percent": 100.0
        * sum(float(row["verification_sar"]) for row in rows)
        / len(rows),
        "paired_audio_cluster_bootstrap_95ci_percentage_points": paired_cluster_interval(
            rows, resamples=bootstrap_resamples, seed=bootstrap_seed
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=25)
    args = parser.parse_args()

    rows = [row for path in args.scored for row in read_jsonl(path)]
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Duplicate sample IDs across scored files")
    required = {
        "sample_id",
        "duration_group",
        "audio_sha256",
        "prior",
        "final",
        "prior_correct",
        "final_correct",
        "triggered",
        "accepted",
        "verification_sar",
    }
    missing = [row["sample_id"] for row in rows if not required.issubset(row)]
    if missing:
        raise RuntimeError(f"Scored rows lack required fields: {missing[:5]}")

    groups = sorted({str(row["duration_group"]) for row in rows})
    result = {
        "by_duration": {
            group: metrics(
                [row for row in rows if str(row["duration_group"]) == group],
                args.bootstrap_resamples,
                args.bootstrap_seed,
            )
            for group in groups
        },
        "pooled": metrics(rows, args.bootstrap_resamples, args.bootstrap_seed),
    }
    if {"2", "5"}.issubset(result["by_duration"]):
        result["average_2_5_accuracy_percent"] = 0.5 * (
            result["by_duration"]["2"]["accuracy_percent"]
            + result["by_duration"]["5"]["accuracy_percent"]
        )
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
