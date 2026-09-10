#!/usr/bin/env python3
"""Separate scorer; never imported by aligned inference code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions = load(args.predictions)
    labels = {item["sample_id"]: str(item["ground_truth"]) for item in load(args.labels)}
    if set(labels) != {item["sample_id"] for item in predictions}:
        raise RuntimeError("Prediction and label IDs differ")
    scored = []
    for item in predictions:
        record = dict(item)
        record["ground_truth"] = labels[item["sample_id"]]
        record["correct"] = item.get("prediction") == record["ground_truth"]
        scored.append(record)
    summary = {"samples": len(scored), "correct": sum(item["correct"] for item in scored), "accuracy": sum(item["correct"] for item in scored) / len(scored), "parse_failures": sum(not item.get("parse_success", True) for item in scored)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
