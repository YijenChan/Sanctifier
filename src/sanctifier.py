#!/usr/bin/env python3
"""Canonical implementation of the Sanctifier pipeline described in the paper.

Only the ``evaluate`` stage reads labels.  The preceding stages implement the
paper's transcript prior, hard cross-view trigger, one budgeted waveform audit,
and evidence-attributed reconciliation in that order.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from core import question_stem, read_jsonl, write_json_atomic, write_jsonl_atomic


CHOICE_DIGITS = ("1", "2", "3", "4")


def keyed(path: Path) -> dict[str, dict]:
    return {row["sample_id"]: row for row in read_jsonl(path)}


def index_path(config: dict) -> Path:
    cache = Path(config["data"]["cache_dir"])
    return Path(config["data"].get("index_path", cache / "sample_index_inference.jsonl"))


def transcript_path(config: dict, audio_sha256: str) -> Path:
    cache = Path(config["data"]["cache_dir"])
    directory = Path(config["data"].get("transcripts_dir", cache / "transcripts"))
    return directory / f"{audio_sha256}.json"


def clean_question(question: str) -> str:
    return question.replace(
        "Provide only the choice number and the statement.", ""
    ).strip()


def answer_options(question: str) -> dict[str, str]:
    return {
        number: text.strip()
        for number, text in re.findall(r"\(([1-4])\)\s*([^\n]+)", question)
    }


def normalize_alphanumeric(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def parse_contract(response: str) -> tuple[str | None, str]:
    choice = re.search(r"CHOICE\s*:\s*\(?([1-4])\)?", response, re.IGNORECASE)
    quote = re.search(r"EVIDENCE\s*:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
    if choice is None or quote is None:
        return None, ""
    evidence = quote.group(1).strip().strip('"').strip("'")
    if evidence.upper() == "NONE":
        evidence = ""
    return choice.group(1), evidence


def quote_is_attributed(quote: str, note: str) -> bool:
    normalized_quote = normalize_alphanumeric(quote)
    return bool(normalized_quote) and normalized_quote in normalize_alphanumeric(note)


def note_is_relevant(note: str) -> bool:
    negative = re.search(
        r"relevance[^a-z]+(none|uncertain|irrelevant)", note.lower()
    )
    return bool(note.strip()) and negative is None


def should_accept_update(
    *,
    triggered: bool,
    note: str,
    prior: str,
    candidate: str | None,
    quote: str,
    require_attribution: bool,
) -> bool:
    attributed = quote_is_attributed(quote, note)
    return bool(
        triggered
        and note_is_relevant(note)
        and candidate in CHOICE_DIGITS
        and candidate != prior
        and (attributed or not require_attribution)
    )


def prior_prompt(tokenizer, transcript: str, question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You answer multiple-choice questions using only the supplied ASR transcript.",
        },
        {
            "role": "user",
            "content": (
                f"ASR TRANSCRIPT:\n{transcript}\n\n"
                f"QUESTION:\n{clean_question(question)}\n\n"
                "Select the best answer. Return only its number in parentheses."
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) + "("


def choice_token_ids(tokenizer) -> list[int]:
    token_ids: list[int] = []
    for digit in CHOICE_DIGITS:
        encoded = tokenizer.encode(digit, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"Choice {digit!r} is not a single token: {encoded}")
        token_ids.append(int(encoded[0]))
    if len(set(token_ids)) != len(CHOICE_DIGITS):
        raise RuntimeError(f"Choice token IDs are not unique: {token_ids}")
    return token_ids


def score_choices(model, tokenizer, prompt: str, token_ids: list[int]) -> list[float]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits[0, -1, token_ids].float()
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(probabilities)) or not np.isclose(
        probabilities.sum(), 1.0, atol=1e-6
    ):
        raise RuntimeError(f"Invalid choice probabilities: {probabilities}")
    return probabilities.tolist()


def choice_prediction(probabilities: list[float]) -> str:
    return str(int(np.argmax(np.asarray(probabilities))) + 1)


def stage_score_prior(config: dict, output: Path) -> None:
    """Implement Eq. (1): normalized next-token scores on the full transcript."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(int(config["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["text_llm"], local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["models"]["text_llm"],
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    token_ids = choice_token_ids(tokenizer)
    rows = []
    started = time.perf_counter()
    for position, sample in enumerate(read_jsonl(index_path(config)), 1):
        transcript = json.loads(
            transcript_path(config, sample["audio_sha256"]).read_text(encoding="utf-8")
        )
        probabilities = score_choices(
            model,
            tokenizer,
            prior_prompt(tokenizer, str(transcript["text"]), sample["question"]),
            token_ids,
        )
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "duration_group": str(sample.get("duration_group", "unknown")),
                "audio_sha256": sample["audio_sha256"],
                "audio_seconds": float(sample["audio_seconds"]),
                "question": sample["question"],
                "choice_probabilities": probabilities,
                "prediction": choice_prediction(probabilities),
            }
        )
        if position % 25 == 0:
            print(json.dumps({"stage": "score-prior", "completed": position}), flush=True)
    write_jsonl_atomic(output / "priors.jsonl", rows)
    write_json_atomic(
        output / "prior_summary.json",
        {
            "samples": len(rows),
            "choice_token_ids": token_ids,
            "wall_seconds": time.perf_counter() - started,
            "labels_read": False,
        },
    )


def select_random_ids(sample_ids: list[str], count: int, seed: int) -> set[str]:
    if count < 0 or count > len(sample_ids):
        raise ValueError(f"random_trigger_count={count} is outside [0, {len(sample_ids)}]")
    ranked = sorted(
        sample_ids,
        key=lambda sample_id: hashlib.sha256(
            f"{seed}:{sample_id}".encode("utf-8")
        ).hexdigest(),
    )
    return set(ranked[:count])


def trigger_decision(
    policy: str,
    *,
    disagreement: bool,
    sparse: bool,
    sample_id: str,
    random_ids: set[str] | None = None,
) -> bool:
    if policy == "cross_view":
        return disagreement or sparse
    if policy == "disagreement_only":
        return disagreement
    if policy == "sparse_only":
        return sparse
    if policy == "random_count":
        return sample_id in (random_ids or set())
    if policy == "all":
        return True
    if policy == "none":
        return False
    raise ValueError(f"Unknown trigger_policy: {policy}")


def bounded_interval(
    *,
    sample_id: str,
    duration: float,
    proposal: dict | None,
    clip_seconds: float,
    sar_cap: float,
    selection_policy: str,
) -> dict:
    length = min(float(clip_seconds), float(sar_cap) * duration)
    if length <= 0:
        raise ValueError("The verification clip length and SAR cap must be positive")
    if selection_policy == "center" or proposal is None:
        midpoint = duration / 2.0
    elif selection_policy == "random":
        seed = int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16], 16)
        midpoint = float(np.random.default_rng(seed).uniform(0.0, duration))
    elif selection_policy == "query":
        midpoint = 0.5 * (float(proposal["start"]) + float(proposal["end"]))
    else:
        raise ValueError(f"Unknown selection_policy: {selection_policy}")
    start = max(0.0, min(duration - length, midpoint - length / 2.0))
    return {
        "start": start,
        "end": start + length,
        "proposal_score": float(proposal.get("score", 0.0)) if proposal else 0.0,
    }


def stage_prepare(config: dict, output: Path) -> None:
    """Implement Eqs. (3)-(4): hard trigger and one bounded interval."""
    samples = read_jsonl(index_path(config))
    full = keyed(Path(config["data"]["full_results"]))
    local = keyed(Path(config["data"]["rag_results"]))
    method = config["method"]
    policy = str(method.get("trigger_policy", "cross_view"))
    random_ids: set[str] | None = None
    if policy == "random_count":
        random_ids = select_random_ids(
            [row["sample_id"] for row in samples],
            int(method["random_trigger_count"]),
            int(method["random_trigger_seed"]),
        )

    rows = []
    for sample in samples:
        sample_id = sample["sample_id"]
        transcript = json.loads(
            transcript_path(config, sample["audio_sha256"]).read_text(encoding="utf-8")
        )
        duration = float(sample["audio_seconds"])
        words_per_second = len(str(transcript.get("text", "")).split()) / duration
        disagreement = full[sample_id]["prediction"] != local[sample_id]["prediction"]
        sparse = words_per_second < float(method["sparse_words_per_second"])
        triggered = trigger_decision(
            policy,
            disagreement=disagreement,
            sparse=sparse,
            sample_id=sample_id,
            random_ids=random_ids,
        )
        proposals = local[sample_id].get("selected_windows", [])
        proposal = max(proposals, key=lambda row: float(row["score"])) if proposals else None
        intervals = []
        if triggered:
            intervals = [
                bounded_interval(
                    sample_id=sample_id,
                    duration=duration,
                    proposal=proposal,
                    clip_seconds=float(method["verification_clip_seconds"]),
                    sar_cap=float(method["verification_sar_cap"]),
                    selection_policy=str(method.get("selection_policy", "query")),
                )
            ]
        sar = sum(row["end"] - row["start"] for row in intervals) / duration
        rows.append(
            {
                "sample_id": sample_id,
                "duration_group": str(sample.get("duration_group", "unknown")),
                "row_index": sample.get("row_index"),
                "audio_sha256": sample["audio_sha256"],
                "audio_seconds": duration,
                "question": sample["question"],
                "global_hypothesis": full[sample_id]["prediction"],
                "local_hypothesis": local[sample_id]["prediction"],
                "cross_view_disagreement": disagreement,
                "transcript_words_per_second": words_per_second,
                "sparse_transcript": sparse,
                "triggered": triggered,
                "selected_intervals": intervals,
                "verification_sar": sar,
            }
        )
    cap = float(method["verification_sar_cap"])
    write_jsonl_atomic(output / "prepared.jsonl", rows)
    write_json_atomic(
        output / "prepare_summary.json",
        {
            "samples": len(rows),
            "trigger_policy": policy,
            "disagreement": sum(row["cross_view_disagreement"] for row in rows),
            "sparse": sum(row["sparse_transcript"] for row in rows),
            "triggered": sum(row["triggered"] for row in rows),
            "sar_violations": sum(row["verification_sar"] > cap + 1e-9 for row in rows),
            "labels_read": False,
        },
    )


def load_audio_model(config: dict):
    vendor_root = config["models"].get("vendor_root")
    if vendor_root:
        root = Path(vendor_root)
        sys.path.insert(0, str(root / "third_party" / "partial-yarn-4e6e13499b91ee7f7ff893ed297b5710babb55f1"))
        sys.path.insert(0, str(root / "_vendor" / "transformers_4_53_2"))
        from models.modeling_qwen2_audio import Qwen2AudioForConditionalGeneration
        from models.processing_qwen2_audio import Qwen2AudioProcessor
    else:
        from transformers import Qwen2AudioForConditionalGeneration
        try:
            from transformers import Qwen2AudioProcessor
        except ImportError:
            from transformers import AutoProcessor as Qwen2AudioProcessor

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        config["models"]["audio_llm"],
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    processor = Qwen2AudioProcessor.from_pretrained(
        config["models"]["audio_llm"], local_files_only=True
    )
    return model, processor


def mono_16khz(audio: np.ndarray, sampling_rate: int) -> tuple[np.ndarray, int]:
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 2:
        values = values.mean(axis=0 if values.shape[0] <= values.shape[1] else 1)
    if values.ndim != 1:
        raise ValueError(f"Expected mono or stereo audio, got shape {values.shape}")
    if sampling_rate != 16000:
        import librosa

        values = librosa.resample(values, orig_sr=sampling_rate, target_sr=16000)
        sampling_rate = 16000
    return np.asarray(values, dtype=np.float32), sampling_rate


def stage_verify(config: dict, output: Path) -> None:
    """Implement Eq. (5): one choice-blind acoustic evidence record."""
    prepared = read_jsonl(output / "prepared.jsonl")
    model, processor = load_audio_model(config)
    data = config["data"]
    table = (
        pq.read_table(Path(data["parquet"]), columns=["audio_array", "sampling_rate"])
        if data.get("parquet")
        else None
    )
    active_filename = None
    active_table = None
    rows = []
    started = time.perf_counter()
    for position, item in enumerate(prepared, 1):
        notes = []
        if item["triggered"]:
            if table is not None:
                audio_row = table.slice(int(item["row_index"]), 1).to_pylist()[0]
            else:
                relative, raw_index = item["sample_id"].split(":", 1)
                filename = relative.split("/", 1)[1]
                if filename != active_filename:
                    active_table = pq.read_table(
                        Path(data["dataset_duration_dir"]) / filename,
                        columns=["audio_array", "sampling_rate"],
                    )
                    active_filename = filename
                audio_row = active_table.slice(int(raw_index), 1).to_pylist()[0]
            audio, sampling_rate = mono_16khz(
                np.asarray(audio_row["audio_array"]), int(audio_row["sampling_rate"])
            )
            for interval in item["selected_intervals"]:
                clip = audio[
                    int(interval["start"] * sampling_rate):
                    int(interval["end"] * sampling_rate)
                ]
                choices_visible = bool(
                    config["method"].get("verifier_choices_visible", False)
                )
                focus = item["question"] if choices_visible else question_stem(item["question"])
                prompt = (
                    "Listen faithfully to this clip. Do not guess or add inaudible details. "
                    f"QUESTION FOCUS:\n{focus}\n\n"
                    "Format exactly: HEARD: <faithful local evidence>; "
                    "RELEVANCE: <relevant/uncertain/none>."
                )
                conversation = [
                    {
                        "role": "system",
                        "content": "You extract local audio evidence without answer choices.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio_array": clip},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
                rendered = processor.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False
                )
                inputs = processor(
                    text=rendered,
                    audio=[clip],
                    return_tensors="pt",
                    padding=True,
                    sampling_rate=sampling_rate,
                ).to("cuda")
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)
                response = processor.batch_decode(
                    generated[:, inputs.input_ids.size(1):],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                notes.append(
                    {
                        **interval,
                        "raw_output": response,
                        "choices_visible": choices_visible,
                    }
                )
        rows.append({"sample_id": item["sample_id"], "verification_notes": notes})
        if position % 10 == 0:
            print(json.dumps({"stage": "verify", "completed": position}), flush=True)
    write_jsonl_atomic(output / "verifications.jsonl", rows)
    write_json_atomic(
        output / "verify_summary.json",
        {
            "samples": len(rows),
            "triggered": sum(bool(row["verification_notes"]) for row in rows),
            "wall_seconds": time.perf_counter() - started,
            "labels_read": False,
        },
    )
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def stage_reconcile(config: dict, output: Path) -> None:
    """Implement Eqs. (6)-(7): attributable update or preserve the prior."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    priors = keyed(output / "priors.jsonl")
    prepared = keyed(output / "prepared.jsonl")
    verifications = keyed(output / "verifications.jsonl")
    if not (set(priors) == set(prepared) == set(verifications)):
        raise RuntimeError("Prior, preparation, and verification sample IDs differ")

    set_seed(int(config["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["text_llm"], local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["models"]["text_llm"],
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    require_attribution = bool(config["method"].get("require_attribution", True))
    rows = []
    started = time.perf_counter()
    for position, (sample_id, prior_row) in enumerate(priors.items(), 1):
        prior = prior_row["prediction"]
        prepared_row = prepared[sample_id]
        notes = verifications[sample_id]["verification_notes"]
        note = "\n".join(row["raw_output"] for row in notes)
        response = ""
        candidate: str | None = prior
        quote = ""
        if prepared_row["triggered"] and notes and note_is_relevant(note):
            transcript = json.loads(
                transcript_path(config, prior_row["audio_sha256"]).read_text(encoding="utf-8")
            )
            prior_statement = answer_options(prior_row["question"]).get(prior, "")
            prompt = (
                "Decide whether the verified audio evidence corrects the transcript-based prior. "
                "Use the transcript for global context and the audio evidence only for the audited interval. "
                "If you change the answer, copy the shortest supporting phrase verbatim from VERIFIED AUDIO EVIDENCE. "
                "If the evidence is insufficient, keep the prior and write EVIDENCE: NONE.\n\n"
                f"TRANSCRIPT:\n{transcript['text']}\n\n"
                f"TRANSCRIPT PRIOR: ({prior}) {prior_statement}\n\n"
                f"VERIFIED AUDIO EVIDENCE:\n{note}\n\n"
                f"QUESTION:\n{prior_row['question']}\n\n"
                "Return exactly two lines:\nCHOICE: <1-4>\nEVIDENCE: <verbatim phrase or NONE>"
            )
            messages = [
                {
                    "role": "system",
                    "content": "You conservatively reconcile transcript and attributed audio evidence.",
                },
                {"role": "user", "content": prompt},
            ]
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(rendered, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            response = tokenizer.decode(
                generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            parsed_candidate, quote = parse_contract(response)
            candidate = parsed_candidate if parsed_candidate is not None else prior

        attributed = quote_is_attributed(quote, note)
        accepted = should_accept_update(
            triggered=prepared_row["triggered"],
            note=note,
            prior=prior,
            candidate=candidate,
            quote=quote,
            require_attribution=require_attribution,
        )
        rows.append(
            {
                "sample_id": sample_id,
                "duration_group": prior_row["duration_group"],
                "audio_sha256": prior_row["audio_sha256"],
                "prior": prior,
                "triggered": prepared_row["triggered"],
                "candidate": candidate,
                "evidence_quote": quote,
                "quote_attributed": attributed,
                "accepted": accepted,
                "final": candidate if accepted else prior,
                "verification_sar": prepared_row["verification_sar"],
                "raw_response": response,
            }
        )
        if position % 25 == 0:
            print(json.dumps({"stage": "reconcile", "completed": position}), flush=True)
    write_jsonl_atomic(output / "decisions.jsonl", rows)
    write_json_atomic(
        output / "reconcile_summary.json",
        {
            "samples": len(rows),
            "triggered": sum(row["triggered"] for row in rows),
            "accepted": sum(row["accepted"] for row in rows),
            "require_attribution": require_attribution,
            "wall_seconds": time.perf_counter() - started,
            "labels_read": False,
        },
    )


def paired_cluster_interval(
    rows: list[dict], *, resamples: int = 10_000, seed: int = 25
) -> list[float]:
    clusters: dict[str, list[float]] = {}
    for row in rows:
        delta = float(row["final_correct"]) - float(row["prior_correct"])
        # Duration is part of the cluster key because the same source hash can
        # occur in more than one benchmark condition.
        cluster = f"{row.get('duration_group', 'unknown')}:{row['audio_sha256']}"
        clusters.setdefault(cluster, []).append(delta)
    keys = sorted(clusters)
    if not keys:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(keys), size=len(keys))
        values = [value for key_index in sampled for value in clusters[keys[key_index]]]
        estimates[index] = 100.0 * float(np.mean(values))
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def stage_evaluate(config: dict, output: Path) -> None:
    """Read labels only here and produce the paired answer/budget diagnostics."""
    decisions = keyed(output / "decisions.jsonl")
    labels = {
        row["sample_id"]: str(row["ground_truth"])
        for row in read_jsonl(Path(config["data"]["labels"]))
    }
    if set(decisions) != set(labels):
        raise RuntimeError("Decision and label sample IDs differ")
    rows = []
    for sample_id, decision in decisions.items():
        label = labels[sample_id]
        rows.append(
            {
                **decision,
                "ground_truth": label,
                "prior_correct": decision["prior"] == label,
                "final_correct": decision["final"] == label,
            }
        )
    prior_correct = sum(row["prior_correct"] for row in rows)
    final_correct = sum(row["final_correct"] for row in rows)
    wrong_to_right = sum(
        (not row["prior_correct"]) and row["final_correct"] for row in rows
    )
    right_to_wrong = sum(
        row["prior_correct"] and (not row["final_correct"]) for row in rows
    )
    try:
        from scipy.stats import binomtest

        discordant = wrong_to_right + right_to_wrong
        mcnemar_p = (
            float(binomtest(min(wrong_to_right, right_to_wrong), discordant, 0.5).pvalue)
            if discordant
            else 1.0
        )
    except ImportError:
        mcnemar_p = None
    sars = [float(row["verification_sar"]) for row in rows]
    result = {
        "samples": len(rows),
        "prior_correct": prior_correct,
        "final_correct": final_correct,
        "prior_accuracy_percent": 100.0 * prior_correct / len(rows),
        "accuracy_percent": 100.0 * final_correct / len(rows),
        "gain_percentage_points": 100.0 * (final_correct - prior_correct) / len(rows),
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "changed": sum(row["prior"] != row["final"] for row in rows),
        "triggered": sum(row["triggered"] for row in rows),
        "accepted": sum(row["accepted"] for row in rows),
        "mean_verification_sar_percent": 100.0 * float(np.mean(sars)),
        "p95_verification_sar_percent": 100.0 * float(np.quantile(sars, 0.95)),
        "max_verification_sar_percent": 100.0 * max(sars),
        "paired_audio_cluster_bootstrap_95ci_percentage_points": paired_cluster_interval(
            rows,
            resamples=int(config["evaluation"].get("bootstrap_resamples", 10_000)),
            seed=int(config["evaluation"].get("bootstrap_seed", 25)),
        ),
        "mcnemar_exact_two_sided_p": mcnemar_p,
    }
    write_jsonl_atomic(output / "scored_results.jsonl", rows)
    write_json_atomic(output / "evaluation.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=["score-prior", "prepare", "verify", "reconcile", "evaluate"],
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path(config["runtime"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    {
        "score-prior": stage_score_prior,
        "prepare": stage_prepare,
        "verify": stage_verify,
        "reconcile": stage_reconcile,
        "evaluate": stage_evaluate,
    }[args.stage](config, output)


if __name__ == "__main__":
    main()
