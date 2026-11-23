#!/usr/bin/env python3
"""Compare generic QA and task-focused evidence prompts on one MVVBench counting case."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from video_evidence import get_runtime, qa_prompt


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def task_dir(root: Path, qid: str) -> Path:
    digest = hashlib.sha256((qid + "all_views_frames_2").encode()).hexdigest()[:18]
    return root / "tasks" / digest


def focused_prompt(question: str, options: dict[str, str]) -> str:
    choices = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"""任务：回答下面关于穿工作背心人数的选择题。所有图像来自同一场景的两个摄像机视角；每个视角有两个时间相近的帧。

问题：{question}
选项：
{choices}

请先整理证据，再选答案：
1. 按视角和帧分别列出直接可见的穿背心者。标注九宫格位置、背心是否清晰可见、遮挡情况，以及简短可核对的外观线索。性别只记录表观判断，不确定时写 uncertain。
2. 比较同一视角的两帧，避免把同一个人重复计数；再比较两个视角。跨视角只有在衣着细节、姿态、相对位置等线索足够时才对应同一人。仅凭都穿橙色背心或方位相似不足以匹配。匹配不确定时明确保留不确定。
3. 检查某人是否在一个视角中被遮挡、而在另一个视角中可见。明确指出每个视角可见的人数与被遮挡线索。不能从被遮挡区域臆测人物或背心状态。
4. 给出跨帧、跨视角去重后的背心穿着者人数和答案。不要把各视角人数直接相加；若无法可靠去重，给出确定下限并说明答案依据与限制。

只输出有效 JSON，不要输出隐藏的逐字思维过程：
{{
  "per_view_frame_evidence": [{{"view": "View 1", "frame": "frame_000.jpg", "visible_vest_wearers": [{{"local_id": "P1", "grid_region": "left_upper|upper_middle|right_upper|left_middle|center|right_middle|left_lower|lower_middle|right_lower|uncertain", "vest": "yes|partial|uncertain", "occlusion": "none|partial|mostly_occluded|uncertain", "apparent_gender": "male_presenting|female_presenting|uncertain", "visible_cues": "..."}}], "count_visible": 0}}],
  "cross_view_matches": [{{"person_refs": ["View 1/frame_000/P1", "View 3/frame_000/P2"], "status": "supported|uncertain|not_matched", "basis": "..."}}],
  "occlusion_complementarity": [{{"person_ref": "...", "view_1": "...", "view_3": "...", "basis": "..."}}],
  "unique_vest_wearer_count": 0,
  "answer": "A|B|C|D",
  "confidence": "low|medium|high",
  "limitations": ["..."]
}}

只报告图像支持的内容。若帧或视角不足以判断精确人数，不要为了选出答案而编造跨视角对应。"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--query-id", default="63am_industry_safety_2-q01180")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reasoning-max-new-tokens", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = args.diagnostic_dir.resolve()
    manifest_path, predictions_path = root / "diagnostic_manifest.json", root / "predictions.json"
    for path in (manifest_path, predictions_path):
        if not path.is_file():
            parser.error(f"Missing input file: {path}")
    if not args.model_path.is_dir():
        parser.error(f"Model directory does not exist: {args.model_path}")
    if args.reasoning_max_new_tokens < 500:
        parser.error("--reasoning-max-new-tokens must be at least 500")

    manifest, predictions = read_json(manifest_path), read_json(predictions_path)
    query = next((x for x in manifest.get("requests", []) if x.get("query_id") == args.query_id), None)
    if query is None:
        parser.error(f"Query not found: {args.query_id}")
    baseline = next((x for x in predictions if x.get("query_id") == args.query_id
                     and x.get("condition_id") == "all_views_frames_2"), None)
    if not baseline or baseline.get("status") != "completed":
        parser.error("Completed all_views_frames_2 prediction is required")
    views = [str(v) for v in baseline.get("selected_views", [])]
    if set(views) != {"1", "3"} or int(baseline.get("frames_per_view", 0)) != 2:
        parser.error(f"Expected two frames each from Views 1 and 3; found views={views}")

    records, image_meta = [], []
    for view in views:
        paths = sorted((task_dir(root, args.query_id) / f"view_{view}").glob("frame_*.jpg"))
        if len(paths) != 2:
            parser.error(f"Expected 2 frames in View {view}, found {len(paths)}")
        for path in paths:
            records.append({"absolute_file": str(path.resolve()), "label": f"View {view}, {path.name}"})
            image_meta.append({"view": view, "frame": path.name, "path": str(path.resolve()),
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    prompts = {
        "generic_qa": (qa_prompt(query["question"], query["options"]), 128),
        "focused_count_evidence": (focused_prompt(query["question"], query["options"]), args.reasoning_max_new_tokens),
    }
    basis = {"query_id": args.query_id, "model_path": str(args.model_path.resolve()),
             "images": image_meta, "conditions": list(prompts),
             "reasoning_max_new_tokens": args.reasoning_max_new_tokens}
    signature = hashlib.sha256(json.dumps(basis, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    output = args.output.resolve()
    results = []
    if output.exists():
        if not args.resume:
            parser.error(f"Output exists; use a new path or --resume: {output}")
        saved = read_json(output)
        if saved.get("signature") != signature:
            parser.error("Existing output has different settings or input frames")
        results = saved.get("results", [])
    done = {r["condition"] for r in results if r.get("status") == "completed"}

    runtime = get_runtime(args.model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"created_at": datetime.now(timezone.utc).isoformat(),
                        "signature": signature, "settings": basis, "runtime": runtime.metadata,
                        "question": query["question"], "options": query["options"],
                        "gold_answer": query.get("correct_answer"),
                        "note": "Gold answer is stored for post-run comparison and is not included in either prompt.",
                        "results": results})
    for condition, (prompt, max_tokens) in prompts.items():
        if condition in done:
            continue
        print(f"[{condition}] scoring {len(records)} identical frames", flush=True)
        runtime_dir = output.parent / f"{output.stem}_runtime_{condition}"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        row = {"condition": condition, "status": "error", "prompt": prompt}
        try:
            raw, metrics = runtime(records, max_tokens, runtime_dir, prompt)
            row.update(status="completed", raw_response=raw, metrics=metrics)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        results = [r for r in results if r.get("condition") != condition] + [row]
        write_json(output, {"created_at": datetime.now(timezone.utc).isoformat(),
                            "signature": signature, "settings": basis, "runtime": runtime.metadata,
                            "question": query["question"], "options": query["options"],
                            "gold_answer": query.get("correct_answer"),
                            "note": "Gold answer is stored for post-run comparison and is not included in either prompt.",
                            "results": results})
        print(json.dumps({"condition": condition, "status": row["status"],
                          "error": row.get("error")}, ensure_ascii=False), flush=True)
    print(f"Saved comparison: {output}", flush=True)
    return 0 if all(r.get("status") == "completed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

