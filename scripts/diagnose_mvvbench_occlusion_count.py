#!/usr/bin/env python3
"""Inspect person visibility, occlusion, and cross-view counting for one MVVBench item."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from video_evidence import get_runtime


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def task_dir(root: Path, query_id: str, condition_id: str) -> Path:
    digest = hashlib.sha256((query_id + condition_id).encode()).hexdigest()[:18]
    return root / "tasks" / digest


def build_prompt() -> str:
    return """这些图像来自同一室内场景的两个摄像机视角，每个视角含两个相邻时间点的画面。请只做人物可见性与遮挡盘点，不回答选择题，也不要假设人物总数。

请逐张图检查穿工作背心的人：
1. 给每张图中可见或部分可见的人临时编号，并记录九宫格位置、背心是否可见、遮挡程度（无遮挡/部分遮挡/大部分遮挡/无法判断），以及支持判断的可见特征。
2. 只根据图像中可见的外观，记录表观性别（男性化呈现/女性化呈现/无法判断）；不要把表观判断说成已知身份。看不清时标记无法判断。
3. 比较同一视角的相邻帧，再比较两个视角。只有有具体外观线索支持时才把人物跨帧或跨视角对应；不要仅凭“都穿橙色背心”就认定是同一个人。无法确认时保留为不同候选或 uncertain。
4. 检查是否有人在一个视角中被遮挡、而在另一个视角中可见。分别说明这是直接观察还是跨视角推断，并描述其位置与遮挡来源；没有证据时不要推断被遮挡者的性别或身份。
5. 最后给出能够从这些图像确认的不同背心穿着者人数；如果不能确定精确人数，给出可确认的下限和不确定原因。跨帧、跨视角重复出现的同一个人只能计一次。

只输出有效 JSON，不要附加 Markdown：
{
  "per_frame_people": [
    {"view": "View 1", "frame": "frame 1", "people": [
      {"person_id": "V1-P1", "vest_visible": "yes|partial|no|uncertain", "apparent_gender": "male_presenting|female_presenting|uncertain", "grid_region": "left_upper|upper_middle|right_upper|left_middle|center|right_middle|left_lower|lower_middle|right_lower|uncertain", "occlusion": "none|partial|mostly_occluded|uncertain", "visible_cues": "..."}
    ]}
  ],
  "cross_view_matches": [
    {"person_ids": ["V1-P1", "V3-P2"], "match_status": "supported|uncertain|not_matched", "basis": "..."}
  ],
  "occlusion_complementarity": [
    {"person_id_or_match": "...", "view_1_status": "...", "view_3_status": "...", "evidence": "...", "certainty": "low|medium|high"}
  ],
  "distinct_vest_wearer_count": {"count": 0, "minimum_confirmed": 0, "confidence": "low|medium|high", "basis": "..."},
  "limitations": ["..."]
}

证据要简短且可核对。不要把同一人的不同帧重复计数，不要把画面外的人当作已观察到的人，也不要为了给出精确人数而补全不确定信息。"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--query-id", default="63am_industry_safety_2-q01180")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    args = parser.parse_args()

    root = args.diagnostic_dir.resolve()
    manifest_path = root / "diagnostic_manifest.json"
    predictions_path = root / "predictions.json"
    if not manifest_path.is_file():
        parser.error(f"Missing diagnostic manifest: {manifest_path}")
    if not predictions_path.is_file():
        parser.error(f"Missing predictions: {predictions_path}")
    if not args.model_path.is_dir():
        parser.error(f"Model directory does not exist: {args.model_path}")
    if args.max_new_tokens < 500:
        parser.error("--max-new-tokens must be at least 500 for the structured inventory")

    manifest = load_json(manifest_path)
    predictions = load_json(predictions_path)
    q = next((row for row in manifest.get("requests", [])
              if row.get("query_id") == args.query_id), None)
    if q is None:
        parser.error(f"Query ID not found in diagnostic manifest: {args.query_id}")
    baseline = next((row for row in predictions
                     if row.get("query_id") == args.query_id
                     and row.get("condition_id") == "all_views_frames_2"), None)
    if baseline is None or baseline.get("status") != "completed":
        parser.error(f"Missing completed all_views_frames_2 result for {args.query_id}")

    views = [str(view) for view in baseline.get("selected_views", [])]
    if len(views) != 2:
        parser.error(f"Expected exactly two selected views, found {views}")
    records: list[dict[str, str]] = []
    image_metadata: list[dict[str, Any]] = []
    frames_per_view = int(baseline.get("frames_per_view", 0))
    if frames_per_view != 2:
        parser.error(f"Expected two frames per view, found {frames_per_view}")
    for view in views:
        folder = task_dir(root, args.query_id, "all_views_frames_2") / f"view_{view}"
        frames = sorted(folder.glob("frame_*.jpg"))
        if len(frames) != frames_per_view:
            parser.error(f"Expected 2 frames for View {view}, found {len(frames)} in {folder}")
        for index, frame in enumerate(frames, start=1):
            records.append({"absolute_file": str(frame.resolve()),
                            "label": f"View {view} frame {index}"})
            image_metadata.append({
                "view": view, "frame": frame.name, "path": str(frame.resolve()),
                "sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
            })

    prompt = build_prompt()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime = get_runtime(args.model_path)
    print(f"Inspecting {args.query_id}: {len(records)} frames from Views {', '.join(views)}", flush=True)
    runtime_dir = output.parent / (output.stem + "_runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    raw, metrics = runtime(records, args.max_new_tokens, runtime_dir, prompt)
    write_json(output, {
        "query_id": args.query_id,
        "question": q.get("question"),
        "views": views,
        "images": image_metadata,
        "prompt": prompt,
        "raw_response": raw,
        "metrics": metrics,
        "runtime": runtime.metadata,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "Apparent gender labels describe visual presentation only; they are not verified identity attributes.",
    })
    print(f"Saved result: {output}", flush=True)
    print(raw, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

