#!/usr/bin/env python3
"""Measure multiple-choice option-position bias on one MVVBench query.

For each available single-view condition, sends the same sampled images and
question in all 24 permutations of the four answer choices. The script maps
each selected display label back to the original option and records whether
the model selected the option currently shown first.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import itertools
import json
import mimetypes
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request


API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl:free"
LABELS = ("A", "B", "C", "D")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_options(value) -> dict[str, str]:
    if isinstance(value, dict):
        options = {str(k).upper(): str(v) for k, v in value.items()}
    elif isinstance(value, list):
        options = {LABELS[i]: str(text) for i, text in enumerate(value[:4])}
    else:
        raise ValueError("Question options must be a list or A-D mapping")
    options = {key: options[key] for key in LABELS if key in options}
    if tuple(options) != LABELS:
        raise ValueError(f"Expected exactly four A-D options, got {list(options)}")
    return options


def image_part(path: Path) -> dict:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}", "detail": "high"}}


def request_model(api_key: str, model: str, question: str, displayed_options: list[str],
                  frames: list[dict], max_tokens: int, timeout: float,
                  reasoning_effort: str) -> dict:
    options_text = "\n".join(f"{label}. {text}" for label, text in zip(LABELS, displayed_options))
    prompt = (
        f"Question: {question}\n\nOptions:\n{options_text}\n\n"
        'Choose the best answer. Return only JSON: '
        '{"answer":"A|B|C|D","answer_text":"exact option text"}. '
        "If the visual evidence is insufficient, choose the closest supported option."
    )
    content = [{"type": "text", "text": prompt}]
    for i, frame in enumerate(frames, start=1):
        content.append({"type": "text", "text": f"\nImage {i} (View {frame.get('view')}):"})
        content.append(image_part(Path(frame["model_input_image"])))
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    req = urllib.request.Request(API_URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/JiabinO/Multi-VLM-Cooperation-on-Video-Retrieval",
        "X-Title": "MVVBench Option Order Bias Probe",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {message[:1500]}") from None
    except Exception as exc:
        raise RuntimeError(f"OpenRouter request failed ({type(exc).__name__}): {exc}") from None
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("OpenRouter returned no choices: " + json.dumps(payload)[:1500])
    choice = choices[0]
    message = choice.get("message") or {}
    return {
        "content": message.get("content"),
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage"),
        "provider": payload.get("provider"),
        "model": payload.get("model"),
        "request_id": payload.get("id"),
    }


def parse_choice(content) -> str | None:
    if not isinstance(content, str):
        return None
    try:
        value = json.loads(content)
        answer = str(value.get("answer", "")).strip().upper()
        if answer in LABELS:
            return answer
    except (json.JSONDecodeError, AttributeError):
        pass
    match = re.search(r"\b([A-D])\b", content.upper())
    return match.group(1) if match else None


def query_signature(query: dict, options: dict, conditions: list[dict], model: str) -> str:
    basis = {
        "query_id": query["query_id"], "question": query["question"], "options": options,
        "model": model,
        "conditions": [{"condition_id": c["condition_id"],
                        "frames": [f.get("model_input_image") for f in c.get("sampled_frames", [])]}
                       for c in conditions],
    }
    return hashlib.sha256(json.dumps(basis, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def summarize(results: list[dict], conditions: list[dict]) -> list[dict]:
    summaries = []
    for condition in conditions:
        cid = condition["condition_id"]
        rows = [r for r in results if r.get("condition_id") == cid and r.get("status") == "completed"]
        usable = [r for r in rows if r.get("selected_display_label") in LABELS]
        count = len(usable)
        first_count = sum(r["selected_display_label"] == "A" for r in usable)
        by_position = {label: sum(r.get("selected_display_label") == label for r in usable)
                       for label in LABELS}
        by_original = {label: sum(r.get("selected_original_option") == label for r in usable)
                       for label in LABELS}
        summaries.append({
            "condition_id": cid,
            "attempted": len([r for r in results if r.get("condition_id") == cid]),
            "valid_responses": count,
            "first_position_picks": first_count,
            "first_position_rate": first_count / count if count else None,
            "display_position_counts": by_position,
            "original_option_counts": by_original,
            "correct_count": sum(r.get("is_correct") is True for r in usable),
            "accuracy": (sum(r.get("is_correct") is True for r in usable) / count) if count else None,
        })
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-report", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--reasoning-effort", choices=("none", "minimal", "low"), default="none")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing partial result file")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        parser.error("OPENROUTER_API_KEY is not set in this shell")

    report = load_json(args.screening_report)
    manifest = load_json(args.manifest)
    screened = next((x for x in report.get("review_queue", []) if x.get("query_id") == args.query_id), None)
    query = next((x for x in manifest.get("requests", []) if x.get("query_id") == args.query_id), None)
    if screened is None or query is None:
        parser.error(f"Query ID not found in both screening report and manifest: {args.query_id}")
    options = normalize_options(query.get("options"))
    conditions = [x for x in screened.get("conditions", [])
                  if str(x.get("condition_id", "")).startswith("single_view_")
                  and x.get("status") == "completed" and x.get("sampled_frames")]
    if not conditions:
        parser.error("No completed single-view conditions with sampled frames found")
    for condition in conditions:
        paths = [Path(f["model_input_image"]) for f in condition["sampled_frames"]
                 if f.get("model_input_image")]
        if not paths or any(not p.is_file() for p in paths):
            parser.error(f"Missing input frame file(s) for {condition['condition_id']}")

    signature = query_signature(query, options, conditions, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    if args.output.exists():
        if not args.resume:
            parser.error(f"Output exists: {args.output}; use --resume to continue")
        saved = load_json(args.output)
        if saved.get("signature") != signature:
            parser.error("Existing output does not match this query/model/input configuration")
        results = saved.get("results", [])
    done = {(r.get("condition_id"), tuple(r.get("permutation", [])))
            for r in results if r.get("status") == "completed"}
    tasks = [(condition, perm) for condition in conditions for perm in itertools.permutations(LABELS)
             if (condition["condition_id"], perm) not in done]
    total = len(conditions) * 24
    print(f"Query {args.query_id}: {len(conditions)} single views; {total} permutations; remaining {len(tasks)}", flush=True)

    for index, (condition, perm) in enumerate(tasks, start=1):
        cid = condition["condition_id"]
        displayed = [options[key] for key in perm]
        frames = condition["sampled_frames"]
        record = {
            "condition_id": cid,
            "view": condition.get("selected_views"),
            "permutation": list(perm),
            "displayed_options": {label: options[key] for label, key in zip(LABELS, perm)},
            "first_position_original_option": perm[0],
            "status": "error",
        }
        print(f"[{index}/{len(tasks)}] {cid} order={'-'.join(perm)}", flush=True)
        started = time.time()
        try:
            response = request_model(api_key, args.model, query["question"], displayed,
                                     frames, args.max_tokens, args.timeout, args.reasoning_effort)
            record.update(response)
            answer = parse_choice(response.get("content"))
            record["selected_display_label"] = answer
            record["selected_original_option"] = perm[LABELS.index(answer)] if answer in LABELS else None
            record["selected_option_text"] = (options[record["selected_original_option"]]
                                               if record["selected_original_option"] else None)
            gold = str(query.get("correct_answer", "")).strip().upper()
            record["is_correct"] = (record["selected_original_option"] == gold) if gold in LABELS else None
            if response.get("finish_reason") == "length":
                record["status"] = "truncated"
            elif answer is None:
                record["status"] = "unparsed"
            else:
                record["status"] = "completed"
        except Exception as exc:
            record["error"] = str(exc)
        record["elapsed_seconds"] = round(time.time() - started, 3)
        results.append(record)
        summaries = summarize(results, conditions)
        output_data = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "query_id": args.query_id,
            "question": query["question"],
            "original_options": options,
            "gold_answer": query.get("correct_answer"),
            "model_requested": args.model,
            "signature": signature,
            "method": "All 24 permutations per single-view condition; temperature=0.",
            "interpretation_note": "First-position rate is a position-bias indicator, not proof by itself. Each prompt uses the same images/question and changes only option ordering.",
            "condition_summaries": summaries,
            "results": results,
        }
        args.output.write_text(json.dumps(output_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": record["status"], "selected": record.get("selected_display_label"),
                          "original_option": record.get("selected_original_option"),
                          "finish_reason": record.get("finish_reason"),
                          "elapsed_seconds": record["elapsed_seconds"]}, ensure_ascii=False), flush=True)

    print(f"Saved results: {args.output}")
    for summary in summarize(results, conditions):
        print(f"{summary['condition_id']}: first-position A picked "
              f"{summary['first_position_picks']}/{summary['valid_responses']}; "
              f"original-option counts={summary['original_option_counts']}; "
              f"accuracy={summary['accuracy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

