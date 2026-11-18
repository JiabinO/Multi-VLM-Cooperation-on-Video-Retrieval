#!/usr/bin/env python3
"""Evaluate single-image person, work-vest, location, and occlusion recognition."""
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
    return """请只分析这张单独的室内场景图像，进行人物视觉盘点。不要假设其他摄像机视角，不要推测画面外或完全被遮住的人，也不要猜测总人数。

请找出图中直接可见或部分可见的人，并逐人记录：
- 是否穿工作背心：yes / no / uncertain。只在背心清晰可见时标 yes；普通衣服颜色相似不能作为背心证据。
- 表观性别：male_presenting / female_presenting / uncertain。只描述图像外观，不代表真实身份；看不清时选 uncertain。
- 画面九宫格位置：left_upper / upper_middle / right_upper / left_middle / center / right_middle / left_lower / lower_middle / right_lower。
- 遮挡：none / partial / mostly_occluded。只对实际可见的身体部分作判断；完全不可见的人不列入人物清单。
- 简短可见特征，帮助人工核对这个人。

同一个人只在这张图中计一次。若人数或背心状态看不清，请给出可确认的下限并说明不确定点。不要输出任何跨视角比较。

只输出有效 JSON：
{
  "visible_people_count": 0,
  "work_vest_wearer_count": 0,
  "minimum_confirmed_people": 0,
  "minimum_confirmed_vest_wearers": 0,
  "people": [
    {"person_id": "P1", "work_vest": "yes|no|uncertain", "apparent_gender": "male_presenting|female_presenting|uncertain", "grid_region": "left_upper|upper_middle|right_upper|left_middle|center|right_middle|left_lower|lower_middle|right_lower|uncertain", "occlusion": "none|partial|mostly_occluded", "visible_cues": "..."}
  ],
  "limitations": ["..."]
}

计数只依据当前图像，不要把未看到的人补进人数。"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--query-id", default="63am_industry_safety_2-q01180")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--resume", action="store_true")
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
    if args.max_new_tokens < 256:
        parser.error("--max-new-tokens must be at least 256")

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
    if len(views) != 2 or set(views) != {"1", "3"}:
        parser.error(f"Expected Views 1 and 3, found {views}")
    frames_per_view = int(baseline.get("frames_per_view", 0))
    if frames_per_view != 2:
        parser.error(f"Expected two frames per view, found {frames_per_view}")

    frames: list[dict[str, Any]] = []
    for view in views:
        folder = task_dir(root, args.query_id, "all_views_frames_2") / f"view_{view}"
        paths = sorted(folder.glob("frame_*.jpg"))
        if len(paths) != frames_per_view:
            parser.error(f"Expected 2 frames for View {view}, found {len(paths)} in {folder}")
        for index, path in enumerate(paths, start=1):
            frames.append({"view": view, "frame": path.name, "index": index,
                           "path": path.resolve(),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    basis = {"query_id": args.query_id, "model_path": str(args.model_path.resolve()),
             "max_new_tokens": args.max_new_tokens,
             "frames": [{"view": f["view"], "frame": f["frame"], "sha256": f["sha256"]}
                        for f in frames]}
    signature = hashlib.sha256(json.dumps(basis, sort_keys=True).encode()).hexdigest()
    output = args.output.resolve()
    results: list[dict[str, Any]] = []
    if output.exists():
        if not args.resume:
            parser.error(f"Output exists; choose a new path or use --resume: {output}")
        saved = load_json(output)
        if saved.get("signature") != signature:
            parser.error("Existing output uses different images or settings")
        results = saved.get("results", [])
    done = {(r.get("view"), r.get("frame")) for r in results
            if r.get("status") == "completed"}

    runtime = get_runtime(args.model_path)
    print(f"Loaded model once; {len(frames) - len(done)} single-image calls pending", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"created_at": datetime.now(timezone.utc).isoformat(),
                        "signature": signature, "settings": basis,
                        "runtime": runtime.metadata,
                        "note": "Each image was submitted independently; no expected count or cross-view evidence was provided.",
                        "results": results})

    for item in frames:
        key = (item["view"], item["frame"])
        if key in done:
            continue
        label = f"View {item['view']} standalone image {item['index']}"
        record = {"absolute_file": str(item["path"]), "label": label}
        runtime_dir = output.parent / f"{output.stem}_runtime_view{item['view']}_{item['index']}"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{item['view']}/{item['index']}] recognize one image", flush=True)
        row: dict[str, Any] = {"view": item["view"], "frame": item["frame"],
                               "image_path": str(item["path"]), "status": "error"}
        try:
            raw, metrics = runtime([record], args.max_new_tokens, runtime_dir, build_prompt())
            row.update(status="completed", raw_response=raw, metrics=metrics)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        results = [r for r in results if (r.get("view"), r.get("frame")) != key]
        results.append(row)
        write_json(output, {"created_at": datetime.now(timezone.utc).isoformat(),
                            "signature": signature, "settings": basis,
                            "runtime": runtime.metadata,
                            "note": "Each image was submitted independently; no expected count or cross-view evidence was provided.",
                            "results": results})
        print(json.dumps({k: row.get(k) for k in
                          ("view", "frame", "status", "error")}, ensure_ascii=False), flush=True)

    print(f"Saved results: {output}", flush=True)
    return 0 if all(r.get("status") == "completed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

