#!/usr/bin/env python3
"""Create a CORAAL-VLD ASR cache and evaluate timestamped evidence retrieval."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def recordings(dataset: Path) -> list[dict]:
    result = []
    for parquet in sorted(dataset.glob("test-*.parquet")):
        result.extend(pq.read_table(parquet, columns=["file_id", "audio"]).to_pylist())
    return result


def qa_records(qa_dir: Path) -> list[dict]:
    result = []
    for path in sorted(qa_dir.glob("*.txt")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            question, times = line.rsplit(":", 1)
            start, end = [float(value.strip()) for value in times.split(",")]
            result.append({
                "sample_id": f"{path.stem}:{line_number}", "file_id": path.stem,
                "question": question.strip(), "answer_start": start, "answer_end": end,
            })
    return result


def transcribe(args: argparse.Namespace) -> None:
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    from faster_whisper.audio import decode_audio

    transcript_dir = args.output / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    model = BatchedInferencePipeline(WhisperModel(str(args.whisper), device="cuda", compute_type="float16"))
    summary, started_all = [], time.perf_counter()
    for position, row in enumerate(recordings(args.dataset), 1):
        path = transcript_dir / f"{row['file_id']}.json"
        if path.exists():
            summary.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        audio = decode_audio(io.BytesIO(row["audio"]["bytes"]), sampling_rate=16000)
        started = time.perf_counter()
        segments, _ = model.transcribe(
            audio, language="en", task="transcribe", beam_size=1, best_of=1,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], vad_filter=True,
            vad_parameters={"max_speech_duration_s": 30}, word_timestamps=False,
            without_timestamps=False, batch_size=4, max_new_tokens=128,
        )
        chunks = [
            {"start": float(segment.start), "end": float(segment.end), "text": segment.text.strip()}
            for segment in segments if segment.text.strip()
        ]
        record = {
            "file_id": row["file_id"], "audio_seconds": len(audio) / 16000.0,
            "asr_seconds": time.perf_counter() - started, "chunks": chunks,
        }
        write_json(path, record)
        summary.append(record)
        print(json.dumps({"status": "transcribe", "completed": position, "total": 14}), flush=True)
    write_json(args.output / "transcription_summary.json", {
        "recordings": len(summary), "audio_seconds": sum(row["audio_seconds"] for row in summary),
        "asr_seconds": sum(row["asr_seconds"] for row in summary),
        "wall_seconds": time.perf_counter() - started_all,
    })


def windows(transcript: dict) -> list[dict]:
    result, start, duration = [], 0.0, float(transcript["audio_seconds"])
    while start < duration:
        end = min(start + 60.0, duration)
        text = " ".join(
            chunk["text"] for chunk in transcript["chunks"]
            if start <= (chunk["start"] + chunk["end"]) / 2.0 < end
        ).strip()
        if text:
            result.append({"start": start, "end": end, "text": text})
        start += 40.0
    return result


def bounded_interval(midpoint: float, length: float, duration: float) -> tuple[float, float]:
    length = min(length, 0.20 * duration)
    start = max(0.0, min(duration - length, midpoint - length / 2.0))
    return start, start + length


def retrieve(args: argparse.Namespace) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(args.bge), device="cuda", local_files_only=True,
                                model_kwargs={"torch_dtype": torch.float16})
    transcript_dir = args.output / "transcripts"
    qas = qa_records(args.qa_dir)
    by_file: dict[str, list[dict]] = {}
    for row in qas:
        by_file.setdefault(row["file_id"], []).append(row)
    rows = []
    for file_id, questions in by_file.items():
        transcript = json.loads((transcript_dir / f"{file_id}.json").read_text(encoding="utf-8"))
        passages = windows(transcript)
        passage_embeddings = model.encode([row["text"] for row in passages], normalize_embeddings=True)
        query_embeddings = model.encode(
            ["Represent this sentence for searching relevant passages: " + row["question"] for row in questions],
            normalize_embeddings=True,
        )
        for question, query in zip(questions, query_embeddings):
            top = passages[int(np.argmax(passage_embeddings @ query))]
            for length in (5.0, 10.0, 20.0, 40.0):
                duration = float(transcript["audio_seconds"])
                seed = int(hashlib.sha256(question["sample_id"].encode()).hexdigest()[:16], 16)
                random_midpoint = np.random.default_rng(seed).uniform(0.0, duration)
                midpoints = {
                    "query": (top["start"] + top["end"]) / 2.0,
                    "random": random_midpoint,
                    "center": duration / 2.0,
                }
                for policy, midpoint in midpoints.items():
                    start, end = bounded_interval(midpoint, length, duration)
                    hit = start < question["answer_end"] and end > question["answer_start"]
                    rows.append({
                        **question, "policy": policy, "clip_cap_seconds": length,
                        "selected_start": start, "selected_end": end,
                        "actual_sar": (end - start) / duration, "hit": hit,
                    })
            window_hit = top["start"] < question["answer_end"] and top["end"] > question["answer_start"]
            rows.append({**question, "policy": "transcript_window", "clip_cap_seconds": 60.0,
                         "selected_start": top["start"], "selected_end": top["end"],
                         "actual_sar": 0.0, "hit": window_hit})
    output = args.output / "retrieval_results.jsonl"
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    aggregates = []
    for policy in ("query", "random", "center", "transcript_window"):
        lengths = (5.0, 10.0, 20.0, 40.0) if policy != "transcript_window" else (60.0,)
        for length in lengths:
            selected = [row for row in rows if row["policy"] == policy and row["clip_cap_seconds"] == length]
            aggregates.append({
                "policy": policy, "clip_cap_seconds": length, "questions": len(selected),
                "recall": sum(row["hit"] for row in selected) / len(selected),
                "mean_actual_sar": sum(row["actual_sar"] for row in selected) / len(selected),
            })
    write_json(args.output / "retrieval_summary.json", {"qa_pairs": len(qas), "runs": aggregates})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["transcribe", "retrieve"])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument("--whisper", type=Path, required=True)
    parser.add_argument("--bge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"transcribe": transcribe, "retrieve": retrieve}[args.stage](args)


if __name__ == "__main__":
    main()
