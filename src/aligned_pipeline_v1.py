#!/usr/bin/env python3
"""Baseline-aligned Sanctifier v1: sparse/disagreement trigger and conservative evidence support."""

from __future__ import annotations

import argparse
import hashlib
import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from core import write_json_atomic, write_jsonl_atomic


ANSWER = ('Your answer MUST be in the format of "(NUMBER) STATEMENT". '
          'For example, if the answer was (4) A pen, you would ONLY output "(4) A pen". '
          'Do NOT include any other text.')
STOP = {"a", "an", "the", "to", "of", "and", "or", "in", "on", "at", "for", "from", "with", "is", "are", "was", "were", "it", "this", "that"}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def keyed(path: Path) -> dict[str, dict]:
    return {item["sample_id"]: item for item in load(path)}


def parse(text: str) -> str | None:
    for pattern in (r"\((\d)\)", r"(\d)\)", r"(\d)\."):
        match = re.search(pattern, text)
        if match and match.group(1) in {"1", "2", "3", "4"}:
            return match.group(1)
    text = text.strip()
    return text[0] if text and text[0] in {"1", "2", "3", "4"} else None


def stem(question: str) -> str:
    return question.split("\n(1)", 1)[0].replace("Provide only the choice number and the statement.", "").strip()


def choices(question: str) -> dict[str, str]:
    return {match.group(1): match.group(2).strip() for match in re.finditer(r"\(([1-4])\)\s*([^\n]+)", question)}


def support(note: str, option: str) -> bool:
    note_tokens = set(re.findall(r"[a-z0-9]+", note.lower()))
    option_tokens = [token for token in re.findall(r"[a-z0-9]+", option.lower()) if token not in STOP and len(token) > 1]
    return bool(option_tokens) and any(token in note_tokens for token in option_tokens)


def stage_prepare(config: dict, output: Path) -> None:
    cache = Path(config["data"]["cache_dir"])
    index = load(Path(config["data"].get("index_path", cache / "sample_index_inference.jsonl")))
    full, rag = keyed(Path(config["data"]["full_results"])), keyed(Path(config["data"]["rag_results"]))
    records = []
    trigger_policy = config["method"].get("trigger_policy", "cross_view")
    random_trigger_ids: set[str] = set()
    if trigger_policy == "random_count":
        count = int(config["method"]["random_trigger_count"])
        seed = int(config["method"]["random_trigger_seed"])
        if count < 0 or count > len(index):
            raise ValueError(f"random_trigger_count={count} is outside [0, {len(index)}]")
        ranked = sorted(
            (sample["sample_id"] for sample in index),
            key=lambda sid: hashlib.sha256(f"{seed}:{sid}".encode("utf-8")).hexdigest(),
        )
        random_trigger_ids = set(ranked[:count])
    for sample in index:
        sid = sample["sample_id"]
        transcript_dir = Path(config["data"].get("transcripts_dir", cache / "transcripts"))
        transcript = json.loads((transcript_dir / f"{sample['audio_sha256']}.json").read_text(encoding="utf-8"))
        density = len(str(transcript.get("text", "")).split()) / float(sample["audio_seconds"])
        disagreement = full[sid]["prediction"] != rag[sid]["prediction"]
        sparse = density < float(config["method"]["sparse_words_per_second"])
        selected = []
        if trigger_policy == "all":
            active = True
        elif trigger_policy == "random_count":
            active = sid in random_trigger_ids
        else:
            active = disagreement or sparse
        if active:
            proposals = rag[sid]["selected_windows"]
            top = max(proposals, key=lambda item: float(item["score"])) if proposals else {"start": 0.0, "end": sample["audio_seconds"], "score": 0.0}
            budget = float(config["method"]["verification_sar_cap"]) * float(sample["audio_seconds"])
            length = min(float(config["method"]["verification_clip_seconds"]), budget)
            selection_policy = config["method"].get("selection_policy", "query")
            if selection_policy == "center":
                midpoint = float(sample["audio_seconds"]) / 2.0
            elif selection_policy == "random":
                random_seed = int(hashlib.sha256(sid.encode("utf-8")).hexdigest()[:16], 16)
                midpoint = float(np.random.default_rng(random_seed).uniform(0.0, float(sample["audio_seconds"])))
            else:
                midpoint = (float(top["start"]) + float(top["end"])) / 2.0
            start = max(0.0, min(float(sample["audio_seconds"]) - length, midpoint - length / 2.0))
            selected = [{"start": start, "end": start + length, "proposal_score": float(top["score"])}]
        records.append({
            "sample_id": sid, "row_index": sample.get("row_index"), "audio_seconds": sample["audio_seconds"],
            "question": sample["question"], "audio_sha256": sample["audio_sha256"],
            "base_raw_response": full[sid]["raw_response"], "base_prediction": full[sid]["prediction"],
            "rag_prediction": rag[sid]["prediction"], "full_rag_disagreement": disagreement,
            "transcript_words_per_second": density, "sparse_trigger": sparse, "selected_intervals": selected,
            "verification_sar": sum(item["end"] - item["start"] for item in selected) / float(sample["audio_seconds"]),
        })
    write_jsonl_atomic(output / "prepared.jsonl", records)
    write_json_atomic(output / "prepare_summary.json", {"samples": len(records), "disagreement": sum(x["full_rag_disagreement"] for x in records), "sparse": sum(x["sparse_trigger"] for x in records), "union_trigger": sum(bool(x["selected_intervals"]) for x in records), "sar_violations": sum(x["verification_sar"] > 0.200001 for x in records)})


def stage_verify(config: dict, output: Path) -> None:
    vendor_root = Path(config["models"]["vendor_root"])
    sys.path.insert(0, str(vendor_root / "third_party" / "partial-yarn-4e6e13499b91ee7f7ff893ed297b5710babb55f1"))
    sys.path.insert(0, str(vendor_root / "_vendor" / "transformers_4_53_2"))
    from models.modeling_qwen2_audio import Qwen2AudioForConditionalGeneration
    from models.processing_qwen2_audio import Qwen2AudioProcessor
    prepared = load(output / "prepared.jsonl")
    table = None
    if config["data"].get("parquet"):
        table = pq.read_table(Path(config["data"]["parquet"]), columns=["audio_array", "sampling_rate"])
    active_file, active_table = None, None
    path = config["models"]["audio_llm"]
    model = Qwen2AudioForConditionalGeneration.from_pretrained(path, local_files_only=True, torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa").eval()
    processor = Qwen2AudioProcessor.from_pretrained(path, local_files_only=True)
    records, started_all = [], time.perf_counter()
    for position, item in enumerate(prepared, 1):
        notes = []
        if item["selected_intervals"]:
            if table is not None:
                row = table.slice(int(item["row_index"]), 1).to_pylist()[0]
            else:
                relative, raw_index = item["sample_id"].split(":", 1)
                filename = relative.split("/", 1)[1]
                if filename != active_file:
                    active_table = pq.read_table(Path(config["data"]["dataset_duration_dir"]) / filename, columns=["audio_array", "sampling_rate"])
                    active_file = filename
                row = active_table.slice(int(raw_index), 1).to_pylist()[0]
            audio, rate = np.asarray(row["audio_array"], dtype=np.float32), int(row["sampling_rate"])
            for interval in item["selected_intervals"]:
                clip = audio[int(interval["start"] * rate):int(interval["end"] * rate)]
                choices_visible = bool(config["method"].get("verifier_choices_visible", False))
                question_focus = item["question"] if choices_visible else stem(item["question"])
                prompt = ("Listen faithfully to this clip. Do not guess or add inaudible details. "
                          f"QUESTION FOCUS:\n{question_focus}\n\n"
                          "Format exactly: HEARD: <faithful local evidence>; RELEVANCE: <relevant/uncertain/none>.")
                conversation = [{"role": "system", "content": "You extract local audio evidence without answer choices."}, {"role": "user", "content": [{"type": "audio", "audio_array": clip}, {"type": "text", "text": prompt}]}]
                text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                inputs = processor(text=text, audio=[clip], return_tensors="pt", padding=True, sampling_rate=rate).to("cuda")
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)
                response = processor.batch_decode(generated[:, inputs.input_ids.size(1):], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
                notes.append({**interval, "raw_output": response, "choices_visible": choices_visible})
        records.append({"sample_id": item["sample_id"], "verification_notes": notes})
        if position % 10 == 0:
            print(json.dumps({"status": "verify_progress", "completed": position}), flush=True)
    write_jsonl_atomic(output / "verifications.jsonl", records)
    write_json_atomic(output / "verify_summary.json", {"samples": len(records), "triggered": sum(bool(x["verification_notes"]) for x in records), "wall_seconds": time.perf_counter() - started_all})


def stage_fuse(config: dict, output: Path) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    cache = Path(config["data"]["cache_dir"])
    prepared, verifications = keyed(output / "prepared.jsonl"), keyed(output / "verifications.jsonl")
    set_seed(int(config["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(config["models"]["text_llm"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(config["models"]["text_llm"], local_files_only=True, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda").eval()
    records = []
    for sid, item in prepared.items():
        transcript_dir = Path(config["data"].get("transcripts_dir", cache / "transcripts"))
        transcript = json.loads((transcript_dir / f"{item['audio_sha256']}.json").read_text(encoding="utf-8"))
        note_records = verifications[sid]["verification_notes"]
        note_text = "\n".join(f"[{n['start']:.1f}-{n['end']:.1f}s] {n['raw_output']}" for n in note_records)
        fusion_response, fusion_prediction = item["base_raw_response"], item["base_prediction"]
        if note_records:
            prompt = ("Use only the ASR transcript and verified raw-audio evidence to answer the multiple-choice question. "
                      "Raw-audio evidence may correct a local ASR error, but ignore it if irrelevant or uncertain.\n"
                      f"ASR TRANSCRIPT:\n{transcript['text']}\n\nFROZEN BASE RESPONSE:\n{item['base_raw_response']}\n\n"
                      f"VERIFIED RAW-AUDIO EVIDENCE:\n{note_text}\n\nQUESTION:\n{item['question']}\n\n{ANSWER}")
            messages = [{"role": "system", "content": "You conservatively reconcile ASR and verified raw-audio evidence."}, {"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            fusion_response = tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            fusion_prediction = parse(fusion_response)
        opts = choices(item["question"])
        relevant = bool(note_records) and not any(marker in note_text.lower() for marker in ["relevance: none", "relevance: uncertain"])
        supported = bool(fusion_prediction and fusion_prediction in opts and support(note_text, opts[fusion_prediction]))
        policies = {}
        for policy in config["method"]["trigger_policies"]:
            active = item["full_rag_disagreement"] if policy == "disagreement_only" else (item["full_rag_disagreement"] or item["sparse_trigger"])
            accept = bool(active and relevant and supported and fusion_prediction and fusion_prediction != item["base_prediction"])
            policies[policy] = fusion_prediction if accept else item["base_prediction"]
        records.append({
            "sample_id": sid, "base_prediction": item["base_prediction"], "rag_prediction": item["rag_prediction"],
            "fusion_raw_response": fusion_response, "fusion_prediction": fusion_prediction,
            "evidence_relevant": relevant, "option_token_supported": supported,
            "predictions": policies, "verification_sar": item["verification_sar"],
            "trigger_disagreement": item["full_rag_disagreement"], "trigger_sparse": item["sparse_trigger"],
            "verification_notes": note_records,
        })
    write_jsonl_atomic(output / "final_results.jsonl", records)


def stage_score(config: dict, output: Path) -> None:
    labels = {item["sample_id"]: str(item["ground_truth"]) for item in load(Path(config["data"]["labels"]))}
    records = load(output / "final_results.jsonl")
    result = {}
    for policy in config["method"]["trigger_policies"]:
        base_correct = [item["base_prediction"] == labels[item["sample_id"]] for item in records]
        final_correct = [item["predictions"][policy] == labels[item["sample_id"]] for item in records]
        result[policy] = {
            "samples": len(records), "base_correct": sum(base_correct), "final_correct": sum(final_correct),
            "accuracy": sum(final_correct) / len(records),
            "wrong_to_right": sum((not b) and f for b, f in zip(base_correct, final_correct)),
            "right_to_wrong": sum(b and (not f) for b, f in zip(base_correct, final_correct)),
            "changed": sum(item["predictions"][policy] != item["base_prediction"] for item in records),
        }
    result["sar"] = {"mean": float(np.mean([item["verification_sar"] for item in records])), "max": max(item["verification_sar"] for item in records), "violations": sum(item["verification_sar"] > 0.200001 for item in records)}
    write_json_atomic(output / "score.json", result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "verify", "fuse", "score"])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path(config["runtime"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    globals()[f"stage_{args.stage}"](config, output)
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
