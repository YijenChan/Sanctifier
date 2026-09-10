"""Pure utilities for the Sanctifier pipeline."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def question_stem(question: str) -> str:
    return (
        question.split("\n(1)", 1)[0]
        .replace("Provide only the choice number and the statement.", "")
        .strip()
    )


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("Non-finite value in normalization input")
    if high - low < 1e-12:
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]


def js_divergence(p: list[float], q: list[float]) -> float:
    left = np.asarray(p, dtype=np.float64)
    right = np.asarray(q, dtype=np.float64)
    left = left / left.sum()
    right = right / right.sum()
    middle = 0.5 * (left + right)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(left, middle) + 0.5 * kl(right, middle)


def build_coarse_windows(
    fine_segments: list[dict],
    duration: float,
    window_seconds: float,
    overlap_seconds: float,
) -> list[dict]:
    stride = window_seconds - overlap_seconds
    if stride <= 0:
        raise ValueError("coarse window overlap must be smaller than window size")
    windows: list[dict] = []
    start = 0.0
    window_id = 0
    while start < duration:
        end = min(duration, start + window_seconds)
        segment_ids = []
        texts = []
        for segment in fine_segments:
            midpoint = 0.5 * (float(segment["start"]) + float(segment["end"]))
            if start <= midpoint < end:
                segment_ids.append(int(segment["segment_id"]))
                if str(segment.get("text", "")).strip():
                    texts.append(str(segment["text"]).strip())
        if texts:
            windows.append(
                {
                    "window_id": window_id,
                    "start": start,
                    "end": end,
                    "segment_ids": segment_ids,
                    "text": " ".join(texts),
                }
            )
        start += stride
        window_id += 1
    return windows


def centered_clip(segment: dict, duration: float, clip_seconds: float) -> tuple[float, float]:
    midpoint = 0.5 * (float(segment["start"]) + float(segment["end"]))
    start = max(0.0, midpoint - clip_seconds / 2.0)
    end = min(duration, start + clip_seconds)
    start = max(0.0, end - clip_seconds)
    return start, end


def select_under_budget(candidates: list[dict], budget_seconds: float) -> list[dict]:
    selected: list[dict] = []
    occupied: list[tuple[float, float]] = []
    used = 0.0
    for candidate in sorted(candidates, key=lambda item: float(item["value_score"]), reverse=True):
        start, end = float(candidate["clip_start"]), float(candidate["clip_end"])
        duration = end - start
        overlaps = any(start < prior_end and end > prior_start for prior_start, prior_end in occupied)
        if overlaps or used + duration > budget_seconds + 1e-6:
            continue
        selected.append(candidate)
        occupied.append((start, end))
        used += duration
    return selected
