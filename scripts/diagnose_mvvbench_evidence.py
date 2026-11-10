"""Oracle-timestamp diagnostics for MVVBench view complementarity and frame density.

This diagnostic intentionally uses the benchmark's timestamp annotations. It is
not a deployable selection policy and must not be compared as if it had the
same information as the timestamp-blind EXP-002 selector.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any

from video_evidence import (answer_options, get_runtime, load_mapping,
                            load_questions, parse_answer, question_id,
                            safe_video_path, selected_views, qa_prompt)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def annotation_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_questions(path)
    result = {}
    for i, row in enumerate(rows):
        qid = question_id(row, i)
        if qid in result:
            raise ValueError(f"Duplicate annotation question ID: {qid}")
        result[qid] = row
    return result


def make_conditions(views: list[str], frame_counts: list[int], ablation_frames: int,
                    modes: list[str]) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    seen = set()

    def add(label: str, view_set: list[str], nframes: int, mode: str) -> None:
        key = (tuple(sorted(view_set)), nframes)
        if key in seen:
            return
        seen.add(key)
        conditions.append({"condition_id": label, "mode": mode,
                           "view_ids": sorted(view_set), "frames_per_view": nframes})

    if "view_ablation" in modes:
        add("all_views", views, ablation_frames, "view_ablation")
        for view in views:
            add(f"single_view_{view}", [view], ablation_frames, "view_ablation")
        for view in views:
            remaining = [v for v in views if v != view]
            if remaining:
                add(f"leave_out_view_{view}", remaining, ablation_frames, "view_ablation")
    if "frame_sweep" in modes:
        for nframes in frame_counts:
            add(f"all_views_frames_{nframes}", views, nframes, "frame_sweep")
    return conditions


def sample_timestamp_frames(video: Path, out_dir: Path, start_s: float, end_s: float,
                            nframes: int, jpeg_quality: int, max_image_side: int,
                            point_window_seconds: float) -> list[dict[str, Any]]:
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError(f"Invalid FPS/frame count: {video}")
        duration = count / fps
        lo, hi = max(0.0, start_s), min(max(0.0, end_s), max(0.0, duration - 1 / fps))
        if hi < lo:
            lo = hi = min(max(start_s, 0.0), max(0.0, duration - 1 / fps))
        if hi - lo < 1 / fps:
            half = point_window_seconds / 2
            center = (lo + hi) / 2
            lo, hi = max(0.0, center - half), min(max(0.0, duration - 1 / fps), center + half)
        times = [lo + (i + 0.5) * (hi - lo) / nframes for i in range(nframes)]
        results = []
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, time_s in enumerate(times):
            frame_index = min(count - 1, max(0, round(time_s * fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"Frame decode failed: {video} at {time_s:.3f}s")
            image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            image.thumbnail((max_image_side, max_image_side), Image.Resampling.LANCZOS)
            target = out_dir / f"frame_{i:03d}.jpg"
            image.save(target, format="JPEG", quality=jpeg_quality, optimize=False)
            results.append({"absolute_file": str(target.resolve()), "time_s": frame_index / fps,
                            "bytes": target.stat().st_size, "frame_index": frame_index})
        return results
    finally:
        cap.release()


def run(config_path: Path, questions_path: Path, output: Path, modes: list[str],
        frame_counts: list[int], ablation_frames: int, point_window_seconds: float,
        resume: bool, max_questions: int | None = None,
        questions_per_domain: int | None = None) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ds = config["dataset"]
    annotations = annotation_index(Path(ds["annotation_path"]))
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError("--questions must be a nonempty evaluation_questions.json list")
    if max_questions is not None:
        if max_questions < 1:
            raise ValueError("--max-questions must be positive")
        questions = questions[:max_questions]
    if questions_per_domain is not None:
        if questions_per_domain < 1:
            raise ValueError("--questions-per-domain must be positive")
        selected, counts = [], {}
        for q in questions:
            domain = q.get("dataset", "unknown")
            if counts.get(domain, 0) < questions_per_domain:
                selected.append(q)
                counts[domain] = counts.get(domain, 0) + 1
        questions = selected
    domains = ds.get("domains", [])
    mapping = load_mapping(Path(ds["video_mapping_path"]), domains)
    root = Path(ds["prepared_video_root"])
    media = config.get("media", {})
    quality, side = int(media.get("jpeg_quality", 80)), int(media.get("max_image_side", 448))
    request_rows = []
    for q in questions:
        qid = q.get("query_id")
        if qid not in annotations:
            raise ValueError(f"Question ID not found in source annotations: {qid}")
        ann = annotations[qid]
        if ann.get("timestamp_start") is None or ann.get("timestamp_end") is None:
            raise ValueError(f"Missing timestamp_start/end for {qid}")
        start, end = float(ann["timestamp_start"]), float(ann["timestamp_end"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError(f"Invalid timestamp interval for {qid}: {start}, {end}")
        sample, domain = ann["sample_name"], q.get("dataset") or ann.get("dataset")
        if not domain:
            matches = [d for (s, d) in mapping if s == sample]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous dataset for {qid}: {matches}")
            domain = matches[0]
        available = mapping.get((sample, domain), [])
        selected = selected_views(ann, available)
        views = [{"view": x["view"], "path": str(safe_video_path(root, x["target"]))} for x in selected]
        for item in views:
            if not Path(item["path"]).is_file():
                raise FileNotFoundError(item["path"])
        conds = make_conditions([x["view"] for x in views], frame_counts, ablation_frames, modes)
        request_rows.append({"query_id": qid, "sample_name": sample, "dataset": domain,
                             "question": q["question"], "options": q.get("options") or answer_options(ann),
                             "correct_answer": q.get("correct_answer") or ann.get("correct_answer"),
                             "timestamp_start": start, "timestamp_end": end,
                             "views": views, "conditions": conds})
    request_signature = hashlib.sha256(json.dumps(request_rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True) if resume else output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "diagnostic_manifest.json"
    pred_path = output / "predictions.partial.json"
    if resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {manifest_path}")
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.get("request_signature") != request_signature:
            raise ValueError("Resume inputs/configuration differ from the saved diagnostic manifest")
        predictions = json.loads(pred_path.read_text(encoding="utf-8")) if pred_path.is_file() else []
        done = {(x["query_id"], x["condition_id"]) for x in predictions}
    else:
        write_json(manifest_path, {"request_signature": request_signature,
                   "timestamp_source": str(Path(ds["annotation_path"])), "modes": modes,
                   "frame_counts": frame_counts, "ablation_frames": ablation_frames,
                   "point_window_seconds": point_window_seconds, "requests": request_rows})
        predictions, done = [], set()
    total = sum(len(q["conditions"]) for q in request_rows)
    print(f"Timestamp-oracle diagnostic: {len(request_rows)} questions, {total} conditions; resume={resume}", flush=True)
    meta = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
            "config": config, "python": sys.version, "request_signature": request_signature}
    write_json(output / "run.json", meta)
    t0 = time.perf_counter()
    runtime = None
    try:
        if len(done) < total:
            runtime = get_runtime(Path(config["model"]["model_path"]))
            meta["runtime"] = runtime.metadata
        view_paths = {q["query_id"]: {v["view"]: Path(v["path"]) for v in q["views"]} for q in request_rows}
        for q in request_rows:
            for cond in q["conditions"]:
                key = (q["query_id"], cond["condition_id"])
                if key in done:
                    continue
                digest = hashlib.sha256((q["query_id"] + cond["condition_id"]).encode()).hexdigest()[:18]
                task = output / "tasks" / digest
                records, nbytes, sampled_frame_info = [], 0, []
                for view in cond["view_ids"]:
                    frame_dir = task / f"view_{view}"
                    sampled = sample_timestamp_frames(view_paths[q["query_id"]][view], frame_dir,
                        q["timestamp_start"], q["timestamp_end"], cond["frames_per_view"],
                        quality, side, point_window_seconds)
                    for frame in sampled:
                        records.append({"absolute_file": frame["absolute_file"],
                            "label": f"View {view}, oracle timestamp={frame['time_s']:.3f}s",
                            "view_id": view, "time_s": frame["time_s"]})
                        nbytes += frame["bytes"]
                        sampled_frame_info.append({"view_id": view, "time_s": frame["time_s"],
                                                   "frame_index": frame["frame_index"], "bytes": frame["bytes"]})
                prompt = qa_prompt(q["question"], q["options"])
                (task / "prompt.txt").write_text(prompt, encoding="utf-8")
                raw, metrics = runtime(records, int(config["model"].get("qa_max_new_tokens", 128)), task, prompt)
                (task / "raw_response.txt").write_text(raw, encoding="utf-8")
                answer, confidence = parse_answer(raw)
                predictions.append({"query_id": q["query_id"], "sample_name": q["sample_name"],
                    "dataset": q["dataset"], "condition_id": cond["condition_id"], "mode": cond["mode"],
                    "selected_views": cond["view_ids"], "frames_per_view": cond["frames_per_view"],
                    "timestamp_start": q["timestamp_start"], "timestamp_end": q["timestamp_end"],
                    "input_frames": len(records), "input_bytes": nbytes, "answer": answer,
                    "sampled_frames": sampled_frame_info,
                    "correct_answer": q["correct_answer"], "correct": answer == q["correct_answer"],
                    "confidence": confidence, "status": "completed", "raw_response": raw, "metrics": metrics})
                write_json(pred_path, predictions)
                done.add(key)
                print(f"{len(done)}/{total} {q['query_id']} {cond['condition_id']}: {answer} ({nbytes} B)", flush=True)
        write_json(output / "predictions.json", predictions)
        meta.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter() - t0)
        return 0
    except BaseException as exc:
        meta.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        meta.setdefault("finished_at", datetime.now(timezone.utc).isoformat())
        meta["elapsed_seconds"] = time.perf_counter() - t0
        write_json(output / "run.json", meta)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True,
                        help="evaluation_questions.json from the matching scoring run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=["view_ablation", "frame_sweep"],
                        default=["view_ablation", "frame_sweep"])
    parser.add_argument("--frame-counts", nargs="+", type=int, default=[1, 2, 4, 8],
                        help="uniformly spaced frames per view for the all-view density sweep")
    parser.add_argument("--ablation-frames", type=int, default=4,
                        help="fixed frames per selected view in the view-ablation sweep")
    parser.add_argument("--point-window-seconds", type=float, default=2.0,
                        help="total window centered on zero-duration timestamp annotations")
    parser.add_argument("--max-questions", type=int,
                        help="run the first N evaluation questions; useful for a pilot")
    parser.add_argument("--questions-per-domain", type=int,
                        help="run the first N questions from each dataset for a balanced pilot")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if any(x < 1 for x in args.frame_counts) or args.ablation_frames < 1:
        parser.error("frame counts must be positive integers")
    if not math.isfinite(args.point_window_seconds) or args.point_window_seconds <= 0:
        parser.error("--point-window-seconds must be positive")
    try:
        return run(args.config, args.questions, args.output, args.modes, args.frame_counts,
                   args.ablation_frames, args.point_window_seconds, args.resume,
                   args.max_questions, args.questions_per_domain)
    except Exception as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

