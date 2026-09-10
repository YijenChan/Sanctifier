#!/usr/bin/env python3
"""Evidence-attributed semantic correction over a frozen transcript prior."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
import yaml


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def keyed(path: Path) -> dict[str, dict]:
    return {row["sample_id"]: row for row in load(path)}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def options(question: str) -> dict[str, str]:
    return {number: text.strip() for number, text in re.findall(r"\(([1-4])\)\s*([^\n]+)", question)}


def parse_contract(response: str) -> tuple[str | None, str | None]:
    choice = re.search(r"CHOICE\s*:\s*\(?([1-4])\)?", response, re.IGNORECASE)
    quote = re.search(r"EVIDENCE\s*:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
    if not choice or not quote:
        return None, None
    evidence = quote.group(1).strip().strip('"').strip("'")
    if evidence.upper() == "NONE":
        evidence = ""
    return choice.group(1), evidence


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def quote_is_attributed(quote: str, note: str) -> bool:
    normalized_quote = normalize(quote)
    return bool(normalized_quote) and normalized_quote in normalize(note)


def note_is_relevant(note: str) -> bool:
    lowered = note.lower()
    return bool(note.strip()) and re.search(r"relevance[^a-z]+(none|uncertain|irrelevant)", lowered) is None


def stage_fuse(config: dict, output: Path) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    views = keyed(Path(config["data"]["view_scores"]))
    prepared = keyed(Path(config["data"]["prepared"]))
    verifications = keyed(Path(config["data"]["verifications"]))
    cache = Path(config["data"]["cache_dir"])
    set_seed(int(config["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(config["models"]["text_llm"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["models"]["text_llm"], local_files_only=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to("cuda").eval()

    rows = []
    started = time.perf_counter()
    for position, (sample_id, view) in enumerate(views.items(), 1):
        prior = view["global_prediction"]
        old = prepared[sample_id]
        notes = verifications[sample_id]["verification_notes"]
        active = bool(notes)
        note = "\n".join(item["raw_output"] for item in notes)
        response, candidate, quote = "", prior, ""
        if notes and note_is_relevant(note):
            transcript = json.loads(
                (cache / "transcripts" / f"{view['audio_sha256']}.json").read_text(encoding="utf-8")
            )
            prior_statement = options(view["question"]).get(prior, "")
            prompt = (
                "Decide whether the verified audio evidence corrects the transcript-based prior. "
                "Use the transcript for global context and the audio evidence only for the audited interval. "
                "If you change the answer, copy the shortest supporting phrase verbatim from VERIFIED AUDIO EVIDENCE. "
                "If the evidence is insufficient, keep the prior and write EVIDENCE: NONE.\n\n"
                f"TRANSCRIPT:\n{transcript['text']}\n\n"
                f"TRANSCRIPT PRIOR: ({prior}) {prior_statement}\n\n"
                f"VERIFIED AUDIO EVIDENCE:\n{note}\n\n"
                f"QUESTION:\n{view['question']}\n\n"
                "Return exactly two lines:\nCHOICE: <1-4>\nEVIDENCE: <verbatim phrase or NONE>"
            )
            messages = [
                {"role": "system", "content": "You conservatively reconcile transcript and attributed audio evidence."},
                {"role": "user", "content": prompt},
            ]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(rendered, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            response = tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            parsed_choice, parsed_quote = parse_contract(response)
            if parsed_choice is not None:
                candidate = parsed_choice
                quote = parsed_quote or ""

        attributed = quote_is_attributed(quote, note)
        require_attribution = bool(config.get("method", {}).get("require_attribution", True))
        accepted = bool(
            active and note_is_relevant(note) and candidate != prior
            and (attributed or not require_attribution)
        )
        rows.append({
            "sample_id": sample_id,
            "prior": prior,
            "triggered": active,
            "candidate": candidate,
            "evidence_quote": quote,
            "quote_attributed": attributed,
            "accepted": accepted,
            "final": candidate if accepted else prior,
            "raw_response": response,
        })
        if position % 25 == 0:
            print(json.dumps({"status": "fuse", "completed": position}), flush=True)
    write_jsonl(output / "decisions.jsonl", rows)
    write_json(output / "fuse_summary.json", {
        "samples": len(rows),
        "triggered": sum(row["triggered"] for row in rows),
        "accepted": sum(row["accepted"] for row in rows),
        "wall_seconds": time.perf_counter() - started,
    })


def stage_evaluate(config: dict, output: Path) -> None:
    rows = keyed(output / "decisions.jsonl")
    labels = {row["sample_id"]: str(row["ground_truth"]) for row in load(Path(config["data"]["labels"]))}
    scored = []
    for sample_id, label in labels.items():
        row = rows[sample_id]
        scored.append({
            **row,
            "label": label,
            "prior_correct": row["prior"] == label,
            "final_correct": row["final"] == label,
        })
    result = {
        "samples": len(scored),
        "prior_correct": sum(row["prior_correct"] for row in scored),
        "final_correct": sum(row["final_correct"] for row in scored),
        "wrong_to_right": sum((not row["prior_correct"]) and row["final_correct"] for row in scored),
        "right_to_wrong": sum(row["prior_correct"] and (not row["final_correct"]) for row in scored),
        "changed": sum(row["prior"] != row["final"] for row in scored),
        "triggered": sum(row["triggered"] for row in scored),
        "accepted": sum(row["accepted"] for row in scored),
    }
    write_json(output / config["runtime"].get("evaluation_file", "evaluation.json"), result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["fuse", "evaluate"], required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path(config["runtime"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    {"fuse": stage_fuse, "evaluate": stage_evaluate}[args.stage](config, output)


if __name__ == "__main__":
    main()
