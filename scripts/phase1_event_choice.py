"""Whole-recording multi-label event-choice screening for Assembly101."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

from video_smoke import write_json
from w02_qa import VIEWS, QwenRuntime

PROMPT_TEMPLATE = """你将看到同一机位覆盖完整录制的一组均匀时间采样帧，按时间顺序排列，并带有输入帧编号和名义时间。
下面列出一组官方动作名称。请选择本机位这段完整录制中有直接画面证据支持的所有事件类型，并逐项给出判断。
只能依据当前输入帧；不能因为选项来自数据集标注就默认它发生了。若画面不足以确认，选 uncertain；若在已采样画面中没有看到直接证据，选 not_observed。not_observed 只表示当前采样未观察到，不等于证明整段视频中没有发生。
每个 visible 判断必须引用一个或多个当前输入帧编号，并描述可见证据；不要只复述选项名称。置信度为 0 到 1 的自评，不是校准概率。
对每个选项都输出一项，不要遗漏或重复 ID。仅输出 JSON：
{{"choices":[{{"option_id":"E001","decision":"visible|not_observed|uncertain","confidence":0.0,"evidence_frame_ids":[],"evidence":"具体画面证据；not_observed/uncertain 可说明采样或视角限制"}}]}}

事件选项（ID 顺序对所有视角相同）：
{options}
"""


def load_catalog(manifest_path, views=VIEWS):
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    events = data.get("events")
    if not isinstance(events, list):
        raise ValueError("Annotation manifest must contain an events list")
    per_view = {view: {} for view in views}
    for event in events:
        if event.get("granularity") != "fine" or event.get("view_id") not in per_view:
            continue
        label = event.get("action_cls")
        if not isinstance(label, str) or not label.strip():
            continue
        view = event["view_id"]
        try:
            start = int(event.get("start_frame_30fps", event.get("event_start_frame_30fps")))
            end = int(event.get("end_frame_30fps", event.get("event_end_frame_30fps")))
        except (TypeError, ValueError):
            raise ValueError(f"Invalid event frame interval for {event.get('event_id', label)}")
        if start < 0 or end < start:
            raise ValueError(f"Invalid event frame interval for {event.get('event_id', label)}")
        per_view[view].setdefault(label.strip(), []).append({
            "event_id": event.get("event_id", ""), "start_frame": start, "end_frame": end,
        })
    labels = sorted(set().union(*(set(per_view[v]) for v in views)), key=str.casefold)
    if not labels:
        raise ValueError("No fine-grained event labels found for the requested views")
    return data, per_view, labels


def build_options(labels, seed=0):
    order = list(labels)
    random.Random(seed).shuffle(order)
    return [{"option_id": f"E{i:03d}", "action_cls": label}
            for i, label in enumerate(order, 1)]


def sample_coverage(events_by_label, sample_times_sec, fps=30):
    """Count sampled timestamps inside each label's official intervals (end retained)."""
    result = {}
    for label, intervals in events_by_label.items():
        hit_times = [t for t in sample_times_sec if any(
            interval["start_frame"] / fps <= t <= interval["end_frame"] / fps
            for interval in intervals
        )]
        if len(hit_times) >= 2:
            status = "at_least_two_sampled_frames"
        elif len(hit_times) == 1:
            status = "one_sampled_frame_only"
        else:
            status = "no_sampled_frame_not_evaluable"
        result[label] = {"sampled_frame_count_in_intervals": len(hit_times),
                         "sampled_times_sec": hit_times, "coverage_status": status}
    return result


def parse_choices(raw, options, records):
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    data = json.loads(text)
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list):
        raise ValueError("Expected choices list")
    expected = {item["option_id"] for item in options}
    valid_frames = {int(item["frame_id"]) for item in records}
    seen = set()
    for choice in choices:
        if not isinstance(choice, dict):
            raise ValueError("Choice entries must be objects")
        option_id = choice.get("option_id")
        if option_id not in expected or option_id in seen:
            raise ValueError(f"Unknown or duplicate option ID: {option_id}")
        seen.add(option_id)
        if choice.get("decision") not in {"visible", "not_observed", "uncertain"}:
            raise ValueError(f"Invalid decision for {option_id}")
        confidence = choice.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError(f"Invalid confidence for {option_id}")
        frame_ids = choice.get("evidence_frame_ids")
        if not isinstance(frame_ids, list) or any(type(i) is not int or i not in valid_frames for i in frame_ids):
            raise ValueError(f"Invalid evidence frame IDs for {option_id}")
        if choice["decision"] == "visible":
            if not frame_ids or not isinstance(choice.get("evidence"), str) or not choice["evidence"].strip():
                raise ValueError(f"Visible choice needs cited frames and evidence: {option_id}")
        elif not isinstance(choice.get("evidence"), str):
            raise ValueError(f"Choice needs an evidence note: {option_id}")
    missing = expected - seen
    if missing:
        raise ValueError(f"Missing option IDs: {sorted(missing)}")
    return data


def video_metadata(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or total <= 0:
            raise RuntimeError(f"Invalid video metadata: {path} (fps={fps}, frames={total})")
        return {"fps": fps, "total_frames": total, "duration_sec": total / fps}
    finally:
        cap.release()


def extract_full(video, output, count, end_sec):
    command = [sys.executable, str(Path(__file__).with_name("video_smoke.py")),
               "--video", str(video), "--start", "0", "--end", f"{end_sec:.6f}",
               "--frames", str(count), "--output", str(output), "--extract-only"]
    result = subprocess.run(command, capture_output=True, text=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "extract.stdout.log").write_text(result.stdout, encoding="utf-8")
    (output / "extract.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Full-video extraction failed for {video}; see {output}/extract.stderr.log")
    data = json.loads((output / "frames.json").read_text(encoding="utf-8"))
    return [dict(frame, absolute_file=str((output / frame["file"]).resolve()),
                 label=f"Frame {frame['frame_id']}, nominal time {frame['nominal_time_sec']:.3f}s")
            for frame in data["frames"]], data


def write_review(out, options, per_view, predictions, coverages, views=VIEWS):
    lines = ["# 整段视频官方事件选项审核表", "",
             "逐机位核对模型选择、置信度和证据帧。官方标注只是参照；not_observed 表示未在本次抽样图中看到，不能解释为整段没有发生。",
             "模型置信度是自评。只有人工确认事件事实和帧证据后，才用于筛选后续 QA 候选。", ""]
    for view in views:
        lines += [f"## {view}", "",
                  f"[原始回答]({view}/raw_response.txt) · [帧预览]({view}/preview.jpg) · [帧清单]({view}/frames.json)", "",
                  "| ID | 官方细粒度标注 | 本视角官方出现 | 抽样覆盖 | 模型选择 | 置信度 | 证据帧 | 人工核验 |", "|---|---|---:|---|---|---:|---|---|"]
        pred_by_id = {x.get("option_id"): x for x in predictions.get(view, {}).get("choices", [])}
        for option in options:
            label = option["action_cls"]
            gold = per_view[view].get(label, [])
            coverage = coverages[view].get(label, {"coverage_status": "not_applicable"})
            prediction = pred_by_id.get(option["option_id"], {})
            lines.append(
                f"| {option['option_id']} | {label} | {'yes' if gold else 'no'} | "
                f"{coverage['coverage_status']} | {prediction.get('decision', 'parse/error')} | "
                f"{prediction.get('confidence', '')} | {prediction.get('evidence_frame_ids', '')} | pending |"
            )
        lines.append("")
    lines += ["## 人工核验记录", "",
              "对每个可疑的正确/错误选择：打开对应机位的预览图和帧清单，确认选项所述动作是否在整段出现、模型证据帧是否真的支持该判断、官方标注是否遗漏或边界不同。优先复核 `no_sampled_frame_not_evaluable` 的项目，不要把抽样没覆盖判为模型识别错误。", ""]
    (out / "review.md").write_text("\n".join(lines), encoding="utf-8")


def git_info():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    status = git("status", "--porcelain")
    return {"code_commit": git("rev-parse", "HEAD"),
            "git_dirty": None if status is None else bool(status)}


def run(args, extractor=extract_full, metadata_fn=video_metadata, runtime_factory=QwenRuntime):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    manifest, per_view, labels = load_catalog(args.manifest)
    options = build_options(labels, args.seed)
    option_text = "\n".join(f"{o['option_id']}: {o['action_cls']}" for o in options)
    prompt = PROMPT_TEMPLATE.format(options=option_text)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    run_info = dict(status="running", experiment_id="Assembly101-whole-recording-event-choice",
                    annotation_fps=30, options_count=len(options), config=config,
                    command=sys.argv, python=sys.version, started_at=datetime.now(timezone.utc).isoformat(),
                    **git_info())
    write_json(out / "run.json", run_info)
    write_json(out / "event_options.json", options)
    (out / "error.log").write_text("", encoding="utf-8")

    videos = {view: args.videos / f"{view}_rgb.mp4" for view in VIEWS}
    metadata, extraction_errors, prepared, frame_info = {}, {}, {}, {}
    for view, path in videos.items():
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            metadata[view] = metadata_fn(path)
        except Exception:
            extraction_errors[view] = traceback.format_exc()
    if len(metadata) != len(VIEWS):
        write_json(out / "error_by_view.json", extraction_errors)
        run_info.update(status="failed", finished_at=datetime.now(timezone.utc).isoformat())
        write_json(out / "run.json", run_info)
        return 1
    end_sec = min(meta["duration_sec"] for meta in metadata.values())
    max_delta = max(abs(meta["duration_sec"] - end_sec) for meta in metadata.values())
    if max_delta > args.max_duration_difference_sec:
        run_info["duration_warning"] = f"Camera duration difference {max_delta:.3f}s exceeds threshold"
    run_info["video_metadata"] = metadata
    run_info["common_duration_sec"] = end_sec

    for view, path in videos.items():
        dest = out / view
        try:
            prepared[view], frame_info[view] = extractor(path, dest, args.frames, end_sec)
            frame_times = [r["nominal_time_sec"] for r in frame_info[view]["frames"]]
            coverage = sample_coverage(per_view[view], frame_times)
            for label in labels:
                coverage.setdefault(label, {"sampled_frame_count_in_intervals": 0,
                                            "sampled_times_sec": [],
                                            "coverage_status": "no_sampled_frame_not_evaluable"})
            (dest / "prompt.txt").write_text(prompt, encoding="utf-8")
            write_json(dest / "event_sample_coverage.json", coverage)
        except Exception:
            extraction_errors[view] = traceback.format_exc()
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "error.log").write_text(extraction_errors[view], encoding="utf-8")
    write_json(out / "error_by_view.json", extraction_errors)
    coverages = {view: json.loads((out / view / "event_sample_coverage.json").read_text(encoding="utf-8"))
                 for view in prepared}
    all_frame_times = {v: [r["nominal_time_sec"] for r in frame_info[v]["frames"]] for v in prepared}
    write_json(out / "sampling_coverage_summary.json", {
        "note": "A class is sample-covered if at least one annotation interval contains one or more uniform input frame times; this is not proof that the action is visually recognizable.",
        "frames_per_view": args.frames, "common_duration_sec": end_sec,
        "frame_times_by_view": all_frame_times,
        "coverage_by_view": coverages,
    })
    # Keep full annotation references out of the model prompt; these are for post-run review only.
    write_json(out / "gold_reference.json", {
        "is_evaluation_gold": False,
        "note": "Official annotation presence is a screening reference, not a validated per-view visibility truth. Missing per-view labels are not conclusive negatives.",
        "per_view_labels": {v: per_view[v] for v in VIEWS},
    })
    if args.extract_only:
        predictions = {}
        status = "extracted"
    else:
        runtime = None
        load_error = None
        try:
            runtime = runtime_factory(args.model)
            run_info["runtime"] = runtime.metadata
        except Exception:
            load_error = traceback.format_exc()
            (out / "error.log").write_text(load_error, encoding="utf-8")
        predictions = {}
        for view in VIEWS:
            dest = out / view
            try:
                if view not in prepared:
                    raise RuntimeError(extraction_errors.get(view, "No prepared input"))
                if load_error:
                    raise RuntimeError(load_error)
                (dest / "frames.json").write_text(
                    json.dumps(frame_info[view], ensure_ascii=False, indent=2), encoding="utf-8")
                raw, metrics = runtime(prepared[view], args.max_new_tokens, dest, prompt=prompt)
                (dest / "raw_response.txt").write_text(raw, encoding="utf-8")
                write_json(dest / "metrics.json", metrics)
                parsed = parse_choices(raw, options, prepared[view])
                write_json(dest / "parsed_response.json", parsed)
                predictions[view] = parsed
                run_info.setdefault("task_status", {})[view] = "completed"
            except Exception:
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
                run_info.setdefault("task_status", {})[view] = "failed"
    if args.extract_only:
        for view in VIEWS:
            if view not in prepared:
                run_info.setdefault("task_status", {})[view] = "failed"
            else:
                run_info.setdefault("task_status", {})[view] = "extracted"
    write_review(out, options, per_view, predictions, coverages)
    task_status = run_info.get("task_status", {})
    if any(value == "failed" for value in task_status.values()):
        status = "partial_failure"
    elif not args.extract_only:
        status = "completed_pending_human_review"
    run_info.update(status=status, finished_at=datetime.now(timezone.utc).isoformat())
    write_json(out / "run.json", run_info)
    print(f"Status: {status}; options: {len(options)}; frames/view: {args.frames}; duration: {end_sec:.2f}s")
    print(f"Output: {out.resolve()}")
    return 1 if status == "partial_failure" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="annotation_manifest.json from phase1_annotations.py")
    parser.add_argument("--videos", required=True, type=Path,
                        help="Directory containing the three full-length synchronized *_rgb.mp4 recordings")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--frames", type=int, default=64,
                        help="Uniform samples across the entire common recording duration")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0, help="Deterministic option order shared across views")
    parser.add_argument("--max-duration-difference-sec", type=float, default=0.5)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--extract-only", action="store_true",
                        help="Prepare full-recording frames and coverage tables without loading the model")
    args = parser.parse_args()
    if args.frames < 2 or args.max_new_tokens < 1 or args.max_duration_difference_sec < 0:
        parser.error("Require frames >= 2, positive max-new-tokens, and nonnegative duration tolerance")
    if not args.extract_only and args.model is None:
        parser.error("--model is required for inference")
    if not args.manifest.is_file():
        parser.error(f"Manifest not found: {args.manifest}")
    if not args.videos.is_dir():
        parser.error(f"Video directory not found: {args.videos}")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
