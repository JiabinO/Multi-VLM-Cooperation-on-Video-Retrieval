#!/usr/bin/env python3
"""Run a matched-prompt pilot on multi-view interference candidates.

For each selected question, reuses the exact frames from all_views_frames_2 and
runs: matched single-view QA, original multi-view QA, same-scene prior only, and
same-scene plus explicit cross-view entity reasoning. Loads the local Qwen model
once for the entire pilot and checkpoints each result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

from video_evidence import get_runtime, qa_prompt


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def task_dir(root: Path, query_id: str, condition_id: str) -> Path:
    digest = hashlib.sha256((query_id + condition_id).encode()).hexdigest()[:18]
    return root / "tasks" / digest


def unpack_answer(raw: str) -> tuple[str | None, str | None, str | None]:
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    try:
        parsed = json.loads(text)
    except Exception as exc:
        return None, None, f"JSON parse failed: {exc}"
    answer = parsed.get("answer") if isinstance(parsed, dict) else None
    confidence = parsed.get("confidence") if isinstance(parsed, dict) else None
    if answer not in ("A", "B", "C", "D"):
        return None, confidence, f"Invalid answer field: {answer!r}"
    if confidence not in ("low", "medium", "high"):
        confidence = None
    return answer, confidence, None


def entity_reasoning_prompt(question: str, options: dict[str, str]) -> str:
    option_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"""已知以下图像来自同一场景的不同摄像机视角。请综合图像回答问题。

问题：{question}
选项：
{option_text}

请按以下步骤分析：
1. 分别记录每个视角中与问题有关的直接可见信息，并标明视角来源。
2. 找出可能在多个视角中出现的同一人物、物体或固定地标。说明具体匹配依据；只凭“都像三脚架/红箱/穿相似衣服”不足以确认同一实体。无法确认时标记 uncertain，不要强行对应。
3. 对可靠匹配的实体，比较它们在各视角中的位置或与其他实体的关系。区分画面左右、人物自身左右和场景空间方向。若问题涉及机位方位，说明从公共实体关系到机位结论的简短、可核查依据；仅凭画面位置不能确定时明确说明。
4. 汇总各视角互补或冲突的证据，再选择答案。不要把推断写成直接观察；若证据不足，仍选最可能选项但降低信心。

只输出有效 JSON：
{{
  "per_view_observations": [{{"view": "View 0", "facts": ["直接可见事实及位置"]}}],
  "cross_view_links": [{{"entity": "...", "matching_basis": "...", "confidence": "low|medium|high", "positions_by_view": {{"View 0": "...", "View 3": "..."}}}}],
  "integration_summary": ["简短说明公共实体或互补证据如何支持答案；区分观察和推断"],
  "answer": "A|B|C|D",
  "confidence": "low|medium|high",
  "limitations": ["..." ]
}}

证据要简短且可核验，不编造未见物体，也不要输出隐藏的逐字思维过程，只提供观察和简要理由。"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True,
                        help="Completed diagnostic_run directory containing manifest, predictions, and tasks")
    parser.add_argument("--predictions", type=Path, default=None,
                        help="Defaults to diagnostic-dir/predictions.json")
    parser.add_argument("--query-ids", nargs="+", required=True,
                        help="Pilot query IDs from the high-value possible-interference list")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qa-max-new-tokens", type=int, default=128)
    parser.add_argument("--reasoning-max-new-tokens", type=int, default=1200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_root = args.diagnostic_dir.resolve()
    manifest_path = run_root / "diagnostic_manifest.json"
    predictions_path = (args.predictions or (run_root / "predictions.json")).resolve()
    if not manifest_path.is_file():
        parser.error(f"Missing diagnostic manifest: {manifest_path}")
    if not predictions_path.is_file():
        parser.error(f"Missing completed predictions: {predictions_path}")
    if not args.model_path.is_dir():
        parser.error(f"Model directory does not exist: {args.model_path}")
    if args.qa_max_new_tokens < 16 or args.reasoning_max_new_tokens < 256:
        parser.error("Token limits are too small for the requested output")

    manifest = load_json(manifest_path)
    prior_predictions = load_json(predictions_path)
    if not isinstance(prior_predictions, list):
        parser.error("Predictions must be a JSON list")
    qmap = {row["query_id"]: row for row in manifest.get("requests", [])}
    pmap = {(row.get("query_id"), row.get("condition_id")): row for row in prior_predictions}

    selected: list[dict[str, Any]] = []
    for qid in args.query_ids:
        q = qmap.get(qid)
        if q is None:
            parser.error(f"Query ID is absent from diagnostic manifest: {qid}")
        all_row = pmap.get((qid, "all_views_frames_2"))
        if not all_row or all_row.get("status") != "completed" or all_row.get("correct") is not False:
            parser.error(f"{qid} is not a completed incorrect all_views_frames_2 case")
        single_rows = [pmap.get((qid, f"single_view_{view}")) for view in
                       sorted({str(v["view"]) for v in q["views"]}, key=lambda v: int(v))]
        single_rows = [row for row in single_rows if row and row.get("status") == "completed" and row.get("correct") is True]
        if not single_rows:
            parser.error(f"{qid} has no correct completed single-view prediction")
        single_rows.sort(key=lambda row: int(row["condition_id"].rsplit("_", 1)[1]))
        chosen_single = single_rows[0]
        views = [str(v) for v in all_row["selected_views"]]
        frames_per_view = int(all_row["frames_per_view"])
        if frames_per_view != 2:
            parser.error(f"{qid} all-view baseline has {frames_per_view} frames/view, expected 2")
        all_images: list[dict[str, Any]] = []
        for view in views:
            view_dir = task_dir(run_root, qid, "all_views_frames_2") / f"view_{view}"
            frames = sorted(view_dir.glob("frame_*.jpg"))
            if len(frames) != frames_per_view:
                parser.error(f"Expected {frames_per_view} frames for {qid} View {view}, found {len(frames)} in {view_dir}")
            all_images.extend({"view": view, "path": str(frame.resolve())} for frame in frames)
        chosen_view = str(chosen_single["condition_id"].rsplit("_", 1)[1])
        selected.append({
            "query_id": qid, "dataset": q["dataset"], "sample_name": q["sample_name"],
            "question": q["question"], "options": q["options"],
            "gold_answer": q["correct_answer"], "views": views,
            "all_view_baseline_answer": all_row.get("answer"),
            "historically_correct_single_views": [r["condition_id"] for r in single_rows],
            "matched_single_view": chosen_view,
            "images": all_images,
        })

    conditions = ("matched_single_view", "baseline_multiview", "same_scene_prior", "entity_link_reasoning")
    basis = {
        "diagnostic_run": str(run_root), "predictions_path": str(predictions_path),
        "model_path": str(args.model_path.resolve()), "queries": [
            {**{k: q[k] for k in ("query_id", "dataset", "sample_name", "question", "options", "gold_answer", "views", "matched_single_view")},
             "images": [{"view": x["view"], "path": x["path"],
                         "sha256": hashlib.sha256(Path(x["path"]).read_bytes()).hexdigest()} for x in q["images"]]}
            for q in selected],
        "conditions": list(conditions), "qa_max_new_tokens": args.qa_max_new_tokens,
        "reasoning_max_new_tokens": args.reasoning_max_new_tokens,
    }
    signature = hashlib.sha256(json.dumps(basis, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    if args.output.exists():
        if not args.resume:
            parser.error(f"Output exists; use a new path or pass --resume: {args.output}")
        saved = load_json(args.output)
        if saved.get("signature") != signature:
            parser.error("Existing output belongs to different queries/images/model/settings")
        results = saved.get("results", [])
    done = {(r["query_id"], r["condition"]) for r in results if r.get("status") == "completed"}

    runtime = get_runtime(args.model_path)
    print(f"Loaded model once; pilot queries={len(selected)}, pending calls={len(selected) * len(conditions) - len(done)}", flush=True)
    for q in selected:
        options = q["options"]
        all_records = [{"absolute_file": str(x["path"]), "label": f"View {x['view']} frame {i + 1}"}
                       for view in q["views"] for i, x in enumerate([f for f in q["images"] if f["view"] == view])]
        single_records = [row for row in all_records if row["label"].startswith(f"View {q['matched_single_view']} ")]
        scene_prompt = qa_prompt(q["question"], options) + (
            "\n已知所有输入图像来自同一场景的不同摄像机视角。请综合各视角的可见证据作答；"
            "不要仅因问题给出同场景前提，就认定相似物体是同一个实体。"
        )
        prompts = {
            "matched_single_view": (single_records, qa_prompt(q["question"], options), args.qa_max_new_tokens),
            "baseline_multiview": (all_records, qa_prompt(q["question"], options), args.qa_max_new_tokens),
            "same_scene_prior": (all_records, scene_prompt, args.qa_max_new_tokens),
            "entity_link_reasoning": (all_records, entity_reasoning_prompt(q["question"], options), args.reasoning_max_new_tokens),
        }
        for condition, (records, prompt, max_tokens) in prompts.items():
            if (q["query_id"], condition) in done:
                continue
            print(f"[{q['query_id']}] {condition}: {len(records)} frames", flush=True)
            key = hashlib.sha256((q["query_id"] + condition).encode()).hexdigest()[:18]
            condition_dir = args.output / "tasks" / key
            condition_dir.mkdir(parents=True, exist_ok=True)
            started = time.time()
            row: dict[str, Any] = {
                "query_id": q["query_id"], "dataset": q["dataset"],
                "condition": condition, "status": "error",
                "input_frames": [{"view": rec["label"].split()[1], "path": rec["absolute_file"]} for rec in records],
                "prompt": prompt,
            }
            try:
                raw, metrics = runtime(records, max_tokens, condition_dir, prompt)
                (condition_dir / "raw_response.txt").write_text(raw, encoding="utf-8")
                answer, confidence, parse_error = unpack_answer(raw)
                row.update(status="completed", raw_response=raw, answer=answer,
                           confidence=confidence, parse_error=parse_error,
                           correct=(answer == q["gold_answer"] if answer else None),
                           correct_answer=q["gold_answer"], metrics=metrics)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["elapsed_seconds"] = round(time.time() - started, 2)
            results = [old for old in results
                       if (old.get("query_id"), old.get("condition")) != (q["query_id"], condition)]
            results.append(row)
            write_json(args.output, {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "signature": signature, "settings": basis,
                "runtime": runtime.metadata,
                "interpretation": [
                    "Candidate selection uses prior results; pilot summaries are descriptive, not population estimates.",
                    "Gold answers are used only after generation and are not included in prompts.",
                    "A prompted answer counts as a supported correction only after human review confirms its cited entity links and inference.",
                ], "candidate_metadata": selected, "results": results,
            })
            print(json.dumps({k: row.get(k) for k in
                              ("query_id", "condition", "status", "answer", "confidence", "correct", "parse_error", "error", "elapsed_seconds")},
                             ensure_ascii=False), flush=True)
            if row["status"] == "completed":
                done.add((q["query_id"], condition))
    print(f"Saved {len(results)} condition outputs to {args.output}", flush=True)
    return 0 if all(r.get("status") == "completed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

