#!/usr/bin/env python3
"""Distributional Sanctifier v2 with label-isolated tuning and evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from core import write_json_atomic, write_jsonl_atomic


DIGITS = ("1", "2", "3", "4")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def keyed(path: Path) -> dict[str, dict]:
    return {item["sample_id"]: item for item in load(path)}


def index_file(config: dict) -> Path:
    return Path(config["data"].get("index_path", Path(config["data"]["cache_dir"]) / "sample_index_inference.jsonl"))


def transcript_file(config: dict, sample: dict) -> Path:
    directory = Path(config["data"].get("transcripts_dir", Path(config["data"]["cache_dir"]) / "transcripts"))
    return directory / f"{sample['audio_sha256']}.json"


def stem(question: str) -> str:
    return question.split("\n(1)", 1)[0].replace(
        "Provide only the choice number and the statement.", ""
    ).strip()


def clean_question(question: str) -> str:
    return question.replace("Provide only the choice number and the statement.", "").strip()


def reader_prompt(tokenizer, context: str, question: str, source: str) -> str:
    if source == "transcript":
        system = "You answer multiple-choice questions using only the supplied ASR transcript."
        user = (
            f"ASR TRANSCRIPT:\n{context}\n\nQUESTION:\n{clean_question(question)}\n\n"
            "Select the best answer. Return only its number in parentheses."
        )
    elif source == "acoustic":
        system = "You answer multiple-choice questions using only the supplied acoustic evidence note."
        user = (
            f"ACOUSTIC EVIDENCE:\n{context}\n\nQUESTION:\n{clean_question(question)}\n\n"
            "Select the best answer. Return only its number in parentheses."
        )
    else:
        raise ValueError(source)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "("


def digit_ids(tokenizer) -> list[int]:
    ids = []
    for digit in DIGITS:
        encoded = tokenizer.encode(digit, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"Choice {digit!r} is not a single token: {encoded}")
        ids.append(int(encoded[0]))
    if len(set(ids)) != 4:
        raise RuntimeError(f"Choice token ids are not unique: {ids}")
    return ids


def score_prompt(model, tokenizer, prompt: str, ids: list[int]) -> list[float]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits[0, -1, ids].float()
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(probabilities)) or not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
        raise RuntimeError(f"Invalid choice probabilities: {probabilities}")
    return probabilities.tolist()


def uncertainty(global_p: list[float], local_p: list[float]) -> float:
    p = np.asarray(global_p, dtype=np.float64)
    q = np.asarray(local_p, dtype=np.float64)
    eps = 1e-12
    p, q = np.clip(p, eps, 1.0), np.clip(q, eps, 1.0)
    midpoint = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / midpoint)) + 0.5 * np.sum(q * np.log(q / midpoint))
    entropy = -np.sum(p * np.log(p))
    value = 0.5 * (js / math.log(2.0) + entropy / math.log(4.0))
    return float(np.clip(value, 0.0, 1.0))


def prediction(probabilities: list[float]) -> str:
    return str(int(np.argmax(np.asarray(probabilities))) + 1)


def local_context(record: dict, fallback: str) -> str:
    windows = sorted(record.get("selected_windows", []), key=lambda item: float(item["start"]))
    if not windows:
        return fallback
    return "\n".join(
        f"[{float(item['start']):.1f}-{float(item['end']):.1f}s] {item['text']}" for item in windows
    )


def selected_interval(sample: dict, rag_record: dict, config: dict) -> dict:
    proposals = rag_record.get("selected_windows", [])
    top = max(proposals, key=lambda item: float(item["score"])) if proposals else {
        "start": 0.0, "end": float(sample["audio_seconds"]), "score": 0.0
    }
    duration = float(sample["audio_seconds"])
    length = min(float(config["method"]["verification_clip_seconds"]),
                 float(config["method"]["verification_sar_cap"]) * duration)
    midpoint = 0.5 * (float(top["start"]) + float(top["end"]))
    start = max(0.0, min(duration - length, midpoint - length / 2.0))
    return {"start": start, "end": start + length, "proposal_score": float(top["score"])}


def load_reader(config: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    set_seed(int(config["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(config["models"]["text_llm"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["models"]["text_llm"], local_files_only=True, torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to("cuda").eval()
    return model, tokenizer, digit_ids(tokenizer)


def stage_preflight(config: dict, output: Path) -> None:
    samples = load(index_file(config))[:2]
    full = keyed(Path(config["data"]["full_results"]))
    model, tokenizer, ids = load_reader(config)
    records = []
    for sample in samples:
        transcript = json.loads(transcript_file(config, sample).read_text(encoding="utf-8"))
        prompt = reader_prompt(tokenizer, str(transcript["text"]), sample["question"], "transcript")
        first = score_prompt(model, tokenizer, prompt, ids)
        second = score_prompt(model, tokenizer, prompt, ids)
        records.append({
            "sample_id": sample["sample_id"], "digit_token_ids": ids, "probabilities": first,
            "deterministic_repeat_max_abs_diff": float(np.max(np.abs(np.asarray(first) - np.asarray(second)))),
            "forced_choice_prediction": prediction(first),
            "legacy_generative_prediction": full[sample["sample_id"]]["prediction"],
        })
    write_json_atomic(output / "preflight.json", {
        "status": "passed", "records": records,
        "probability_sums": [sum(item["probabilities"]) for item in records]
    })


def stage_score_views(config: dict, output: Path) -> None:
    samples = load(index_file(config))
    full, rag = keyed(Path(config["data"]["full_results"])), keyed(Path(config["data"]["rag_results"]))
    model, tokenizer, ids = load_reader(config)
    records, started = [], time.perf_counter()
    for position, sample in enumerate(samples, 1):
        sid = sample["sample_id"]
        transcript = json.loads(transcript_file(config, sample).read_text(encoding="utf-8"))
        global_p = score_prompt(model, tokenizer,
                                reader_prompt(tokenizer, str(transcript["text"]), sample["question"], "transcript"), ids)
        local_p = score_prompt(model, tokenizer,
                               reader_prompt(tokenizer, local_context(rag[sid], str(transcript["text"])),
                                             sample["question"], "transcript"), ids)
        density = len(str(transcript.get("text", "")).split()) / float(sample["audio_seconds"])
        interval = selected_interval(sample, rag[sid], config)
        records.append({
            "sample_id": sid, "row_index": sample.get("row_index"), "audio_sha256": sample["audio_sha256"],
            "audio_seconds": float(sample["audio_seconds"]), "question": sample["question"],
            "global_p": global_p, "local_p": local_p,
            "global_prediction": prediction(global_p), "local_prediction": prediction(local_p),
            "uncertainty": uncertainty(global_p, local_p), "transcript_words_per_second": density,
            "selected_interval": interval,
            "legacy_global_prediction": full[sid]["prediction"], "legacy_local_prediction": rag[sid]["prediction"],
        })
        if position % 10 == 0:
            print(json.dumps({"status": "score_views", "completed": position}), flush=True)
    write_jsonl_atomic(output / "view_scores.jsonl", records)
    write_json_atomic(output / "view_score_summary.json", {
        "samples": len(records),
        "global_legacy_agreement": sum(x["global_prediction"] == x["legacy_global_prediction"] for x in records) / len(records),
        "local_legacy_agreement": sum(x["local_prediction"] == x["legacy_local_prediction"] for x in records) / len(records),
        "uncertainty_quantiles": {str(q): float(np.quantile([x["uncertainty"] for x in records], q)) for q in [0, .25, .5, .75, .9, .95, 1]},
        "wall_seconds": time.perf_counter() - started,
    })


def same_interval(a: dict, b: dict, tolerance: float = 1e-4) -> bool:
    return abs(float(a["start"]) - float(b["start"])) <= tolerance and abs(float(a["end"]) - float(b["end"])) <= tolerance


def candidate_trigger(item: dict, config: dict, fixed: dict | None = None) -> bool:
    if fixed is None:
        delta = min(float(x) for x in config["method"]["uncertainty_threshold_grid"])
    else:
        delta = float(fixed["uncertainty_threshold"])
    return item["uncertainty"] > delta or item["transcript_words_per_second"] < float(config["method"]["sparse_words_per_second"])


def stage_verify(config: dict, output: Path) -> None:
    views = load(output / "view_scores.jsonl")
    fixed = None
    if config["data"].get("selected_config"):
        fixed = json.loads(Path(config["data"]["selected_config"]).read_text(encoding="utf-8"))
    reuse = keyed(Path(config["data"]["reuse_verifications"])) if config["data"].get("reuse_verifications") else {}
    records = []
    missing = []
    for item in views:
        notes = []
        if candidate_trigger(item, config, fixed):
            prior = reuse.get(item["sample_id"], {}).get("verification_notes", [])
            if prior and same_interval(item["selected_interval"], prior[0]):
                notes = prior
            else:
                missing.append(item)
        records.append({"sample_id": item["sample_id"], "verification_notes": notes,
                        "reused": bool(notes), "candidate_trigger": candidate_trigger(item, config, fixed)})
    if missing:
        vendor_root = Path(config["models"]["vendor_root"])
        sys.path.insert(0, str(vendor_root / "third_party" / "partial-yarn-4e6e13499b91ee7f7ff893ed297b5710babb55f1"))
        sys.path.insert(0, str(vendor_root / "_vendor" / "transformers_4_53_2"))
        from models.modeling_qwen2_audio import Qwen2AudioForConditionalGeneration
        from models.processing_qwen2_audio import Qwen2AudioProcessor
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            config["models"]["audio_llm"], local_files_only=True, torch_dtype=torch.float16,
            device_map="auto", attn_implementation="sdpa"
        ).eval()
        processor = Qwen2AudioProcessor.from_pretrained(config["models"]["audio_llm"], local_files_only=True)
        table = pq.read_table(Path(config["data"]["parquet"]), columns=["audio_array", "sampling_rate"]) if config["data"].get("parquet") else None
        active_file, active_table = None, None
        by_id = {item["sample_id"]: item for item in records}
        for position, item in enumerate(missing, 1):
            if table is not None:
                row = table.slice(int(item["row_index"]), 1).to_pylist()[0]
            else:
                relative, raw_index = item["sample_id"].split(":", 1)
                filename = relative.split("/", 1)[1]
                if filename != active_file:
                    active_table = pq.read_table(Path(config["data"]["dataset_duration_dir"]) / filename,
                                                 columns=["audio_array", "sampling_rate"])
                    active_file = filename
                row = active_table.slice(int(raw_index), 1).to_pylist()[0]
            audio, rate = np.asarray(row["audio_array"], dtype=np.float32), int(row["sampling_rate"])
            interval = item["selected_interval"]
            clip = audio[int(interval["start"] * rate):int(interval["end"] * rate)]
            prompt = ("Listen faithfully to this clip. Answer choices are hidden. Do not guess or add inaudible details. "
                      f"QUESTION FOCUS:\n{stem(item['question'])}\n\n"
                      "Format exactly: HEARD: <faithful local evidence>; RELEVANCE: <relevant/uncertain/none>.")
            conversation = [{"role": "system", "content": "You extract local audio evidence without answer choices."},
                            {"role": "user", "content": [{"type": "audio", "audio_array": clip},
                                                           {"type": "text", "text": prompt}]}]
            rendered = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=rendered, audio=[clip], return_tensors="pt", padding=True, sampling_rate=rate).to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            response = processor.batch_decode(generated[:, inputs.input_ids.size(1):], skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)[0].strip()
            by_id[item["sample_id"]]["verification_notes"] = [{**interval, "raw_output": response, "choices_visible": False}]
            if position % 10 == 0:
                print(json.dumps({"status": "verify", "completed_new": position, "total_new": len(missing)}), flush=True)
        records = [by_id[item["sample_id"]] for item in views]
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    write_jsonl_atomic(output / "verifications.jsonl", records)
    write_json_atomic(output / "verify_summary.json", {
        "samples": len(records), "candidate_triggered": sum(x["candidate_trigger"] for x in records),
        "reused": sum(x["reused"] for x in records), "new": len(missing)
    })


def relevant(note: str) -> bool:
    lowered = note.lower()
    negative = re.search(r"relevance[^a-z]+(none|uncertain|irrelevant)", lowered)
    return bool(note.strip()) and negative is None


def stage_score_acoustic(config: dict, output: Path) -> None:
    views, notes = keyed(output / "view_scores.jsonl"), keyed(output / "verifications.jsonl")
    model, tokenizer, ids = load_reader(config)
    records = []
    for position, sid in enumerate(views, 1):
        raw = "\n".join(item["raw_output"] for item in notes[sid]["verification_notes"])
        acoustic_p = None
        if relevant(raw):
            acoustic_p = score_prompt(model, tokenizer, reader_prompt(tokenizer, raw, views[sid]["question"], "acoustic"), ids)
        records.append({"sample_id": sid, "raw_note": raw, "relevant": relevant(raw),
                        "acoustic_p": acoustic_p,
                        "acoustic_prediction": prediction(acoustic_p) if acoustic_p else None})
        if position % 10 == 0:
            print(json.dumps({"status": "score_acoustic", "completed": position}), flush=True)
    write_jsonl_atomic(output / "acoustic_scores.jsonl", records)


def decide(view: dict, acoustic: dict, delta: float, kappa: float, sparse_threshold: float) -> dict:
    base = view["global_prediction"]
    triggered = view["uncertainty"] > delta or view["transcript_words_per_second"] < sparse_threshold
    candidate, gain, accepted = base, None, False
    cross_view_agreement = False
    if triggered and relevant(acoustic.get("raw_note", "")) and acoustic["acoustic_p"] is not None:
        joint = np.log(np.clip(np.asarray(view["global_p"]), 1e-12, 1.0)) + np.log(np.clip(np.asarray(acoustic["acoustic_p"]), 1e-12, 1.0))
        candidate = str(int(np.argmax(joint)) + 1)
        gain = float(joint[int(candidate) - 1] - joint[int(base) - 1])
        local_choice = prediction(view["local_p"])
        acoustic_choice = prediction(acoustic["acoustic_p"])
        cross_view_agreement = candidate == local_choice == acoustic_choice
        accepted = candidate != base and cross_view_agreement and gain > kappa
    return {"base": base, "candidate": candidate, "gain": gain, "triggered": triggered,
            "cross_view_agreement": cross_view_agreement,
            "accepted": accepted, "final": candidate if accepted else base}


def metrics(views: dict, acoustics: dict, labels: dict, delta: float, kappa: float, sparse: float) -> dict:
    rows = []
    for sid, view in views.items():
        decision = decide(view, acoustics[sid], delta, kappa, sparse)
        label = labels[sid]
        rows.append({**decision, "sample_id": sid, "label": label,
                     "base_correct": decision["base"] == label,
                     "final_correct": decision["final"] == label})
    return {
        "uncertainty_threshold": delta, "evidence_margin": kappa, "samples": len(rows),
        "base_correct": sum(x["base_correct"] for x in rows),
        "final_correct": sum(x["final_correct"] for x in rows),
        "accuracy": sum(x["final_correct"] for x in rows) / len(rows),
        "wrong_to_right": sum((not x["base_correct"]) and x["final_correct"] for x in rows),
        "right_to_wrong": sum(x["base_correct"] and (not x["final_correct"]) for x in rows),
        "changed": sum(x["final"] != x["base"] for x in rows),
        "triggered": sum(x["triggered"] for x in rows),
        "rows": rows,
    }


def stage_tune(config: dict, output: Path) -> None:
    views, acoustics = keyed(output / "view_scores.jsonl"), keyed(output / "acoustic_scores.jsonl")
    labels = {x["sample_id"]: str(x["ground_truth"]) for x in load(Path(config["data"]["labels"]))}
    sparse = float(config["method"]["sparse_words_per_second"])
    grid = []
    for delta in config["method"]["uncertainty_threshold_grid"]:
        for kappa in config["method"]["evidence_margin_grid"]:
            result = metrics(views, acoustics, labels, float(delta), float(kappa), sparse)
            result.pop("rows")
            grid.append(result)
    best = sorted(grid, key=lambda x: (-x["final_correct"], x["right_to_wrong"], x["triggered"],
                                      -x["evidence_margin"], -x["uncertainty_threshold"]))[0]
    write_json_atomic(output / "tuning_grid.json", {"selection_rule": "registered", "runs": grid})
    write_json_atomic(output / "selected_config.json", best)


def stage_evaluate(config: dict, output: Path) -> None:
    fixed = json.loads(Path(config["data"]["selected_config"]).read_text(encoding="utf-8"))
    views, acoustics = keyed(output / "view_scores.jsonl"), keyed(output / "acoustic_scores.jsonl")
    labels = {x["sample_id"]: str(x["ground_truth"]) for x in load(Path(config["data"]["labels"]))}
    result = metrics(views, acoustics, labels, float(fixed["uncertainty_threshold"]),
                     float(fixed["evidence_margin"]), float(config["method"]["sparse_words_per_second"]))
    intervals = [views[x["sample_id"]]["selected_interval"] for x in result["rows"] if x["triggered"]]
    sars = [(float(v["end"]) - float(v["start"])) / views[row["sample_id"]]["audio_seconds"]
            for row, v in zip([x for x in result["rows"] if x["triggered"]], intervals)]
    # Report both the dataset-level access cost (zero for untriggered examples)
    # and the conditional clip ratio.  The former is the quantity used for
    # system-level accuracy--efficiency comparisons; the latter checks how
    # closely triggered examples approach the per-item cap.
    all_sars = sars + [0.0] * (result["samples"] - len(sars))
    result["verification_sar"] = {
        "mean": float(np.mean(all_sars)) if all_sars else 0.0,
        "p50": float(np.quantile(all_sars, .50)) if all_sars else 0.0,
        "p95": float(np.quantile(all_sars, .95)) if all_sars else 0.0,
        "max": max(all_sars) if all_sars else 0.0,
        "triggered_mean": float(np.mean(sars)) if sars else 0.0,
        "violations": sum(x > float(config["method"]["verification_sar_cap"]) + 1e-9
                          for x in sars),
    }
    rows = result.pop("rows")
    write_jsonl_atomic(output / "final_results.jsonl", rows)
    write_json_atomic(output / "evaluation.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["preflight", "score-views", "verify", "score-acoustic", "tune", "evaluate"], required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path(config["runtime"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    {
        "preflight": stage_preflight,
        "score-views": stage_score_views,
        "verify": stage_verify,
        "score-acoustic": stage_score_acoustic,
        "tune": stage_tune,
        "evaluate": stage_evaluate,
    }[args.stage](config, output)


if __name__ == "__main__":
    main()
