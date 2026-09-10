#!/usr/bin/env python3
"""Frozen-prompt label-free ASR Full/RAG reader for aligned development."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


QUERY = "Represent this sentence for searching relevant passages: "
ANSWER = ('Your answer MUST be in the format of "(NUMBER) STATEMENT". '
          'For example, if the answer was (4) A pen, you would ONLY output "(4) A pen". '
          'Do NOT include any other text.')


def parse_answer(text: str) -> str | None:
    for pattern in (r"\((\d)\)", r"(\d)\)", r"(\d)\."):
        match = re.search(pattern, text)
        if match and match.group(1) in {"1", "2", "3", "4"}:
            return match.group(1)
    stripped = text.strip()
    return stripped[0] if stripped and stripped[0] in {"1", "2", "3", "4"} else None


def windows(transcript: dict) -> list[dict]:
    result, start, duration = [], 0.0, float(transcript["audio_seconds"])
    while start < duration:
        end = min(start + 60.0, duration)
        text = " ".join(
            str(chunk["text"]).strip() for chunk in transcript.get("chunks", [])
            if start <= (float(chunk["start"]) + float(chunk["end"])) / 2.0 < end
        ).strip()
        if text:
            result.append({"start": start, "end": end, "text": text})
        start += 40.0
    return result or [{"start": 0.0, "end": duration, "text": str(transcript.get("text", ""))}]


def stem(question: str) -> str:
    return question.split("\n(1)", 1)[0].replace("Provide only the choice number and the statement.", "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--reader-path", type=Path, required=True)
    parser.add_argument("--bge-path", type=Path)
    parser.add_argument("--variant", choices=["full", "rag"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.index_path or (args.cache_dir / "sample_index_inference.jsonl")
    index = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    set_seed(25)
    tokenizer = AutoTokenizer.from_pretrained(args.reader_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.reader_path, local_files_only=True, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda").eval()
    retriever = None
    if args.variant == "rag":
        retriever = SentenceTransformer(str(args.bge_path), device="cuda", local_files_only=True, model_kwargs={"torch_dtype": torch.float16})
    passage_cache = {}
    records = []
    started_all = time.perf_counter()
    for position, sample in enumerate(index, 1):
        transcript = json.loads((args.cache_dir / "transcripts" / f"{sample['audio_sha256']}.json").read_text(encoding="utf-8"))
        selected = []
        if args.variant == "full":
            context = str(transcript["text"])
        else:
            digest = sample["audio_sha256"]
            if digest not in passage_cache:
                passages = windows(transcript)
                embeddings = retriever.encode([item["text"] for item in passages], normalize_embeddings=True, convert_to_numpy=True)
                passage_cache[digest] = passages, embeddings
            passages, embeddings = passage_cache[digest]
            query = retriever.encode([QUERY + stem(sample["question"])], normalize_embeddings=True, convert_to_numpy=True)[0]
            scores = embeddings @ query
            chosen = np.argsort(-scores)[: min(3, len(passages))]
            selected = [dict(passages[int(i)], score=float(scores[int(i)])) for i in chosen]
            selected.sort(key=lambda item: item["start"])
            context = "\n".join(f"[{item['start']:.1f}-{item['end']:.1f}s] {item['text']}" for item in selected)
        prompt = f"Use only the following ASR transcript to answer the multiple-choice question.\nTRANSCRIPT:\n{context}\n\nQUESTION:\n{sample['question']}\n\n{ANSWER}"
        messages = [{"role": "system", "content": "You answer questions from ASR transcripts."}, {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        elapsed = time.perf_counter() - started
        response = tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        records.append({
            "sample_id": sample["sample_id"],
            "duration_group": str(sample.get("duration_group", "unknown")),
            "variant": args.variant,
            "status": "completed", "raw_response": response, "prediction": parse_answer(response),
            "parse_success": parse_answer(response) is not None, "reader_seconds": elapsed,
            "input_tokens": int(inputs.input_ids.shape[1]), "selected_windows": selected,
        })
        if position % 25 == 0:
            print(json.dumps({"status": "progress", "variant": args.variant, "completed": position}), flush=True)
    output = args.output_dir / "results.jsonl"
    output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")
    summary = {"status": "completed", "variant": args.variant, "samples": len(records), "parse_failures": sum(not item["parse_success"] for item in records), "wall_seconds": time.perf_counter() - started_all, "labels_read": False}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
