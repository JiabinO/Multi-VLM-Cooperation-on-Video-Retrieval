"""Run frozen development questions over five image-view conditions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import sys
import time
import traceback

from video_smoke import write_json
from w02_qa import GROUPS, VIEWS, QwenRuntime, extract, git_info
from phase1_observe import WINDOWS


def prompt_for(question):
    return ("这些采样图片的标签给出了摄像机与时间。只根据可见证据，用中文简短回答；"
            "无法判断时明确说明，不猜测未看见的细节。问题：" + question["question"])


def load_config(path, selected=None):
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions must be a nonempty list")
    ids = set()
    for q in questions:
        if not re.fullmatch(r"W0[1-6]-Q[1-9][0-9]*", q["id"]):
            raise ValueError("Invalid question ID")
        if q["id"] in ids or q["window"] not in WINDOWS:
            raise ValueError("Duplicate ID or invalid window")
        ids.add(q["id"])
        if q["source_start_sec"] != WINDOWS[q["window"]]:
            raise ValueError("Unexpected clip source offset")
        for key in ("question", "reference_answer"):
            if not isinstance(q[key], str) or not q[key].strip():
                raise ValueError(f"Empty {key}")
    if selected:
        if not set(selected) <= ids or len(selected) != len(set(selected)):
            raise ValueError("Unknown or duplicate selected question IDs")
        questions = [q for q in questions if q["id"] in selected]
    return data, questions


def run(args, extractor=extract, runtime_factory=QwenRuntime):
    data, questions = load_config(args.config, args.question_ids)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    meta = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(),
                config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                command=sys.argv, python=sys.version, **git_info(),
                config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    write_json(out / "questions.snapshot.json", data)
    write_json(out / "run.json", meta)
    (out / "error.log").write_text("", encoding="utf-8")
    rows, sampled, errors = [], {}, {}
    runtime, load_error = None, None
    try:
        for window in dict.fromkeys(q["window"] for q in questions):
            for view in VIEWS:
                key = f"{window}_{view}"
                dest = out / "views" / key
                try:
                    video = args.clips / f"{key}.mp4"
                    if not video.is_file():
                        raise FileNotFoundError(video)
                    frames = extractor(video, dest, args.frames, clip_start=WINDOWS[window])["frames"]
                    if not frames:
                        raise ValueError("No sampled frames")
                    sampled[key] = [dict(f, view=view, window=window,
                        absolute_file=str((dest / f["file"]).resolve()),
                        label=f"View {view}, frame {f['frame_id']}, source nominal time={f['nominal_time_sec']:.3f}s")
                        for f in frames]
                except Exception:
                    errors[key] = traceback.format_exc()
                    dest.mkdir(parents=True, exist_ok=True)
                    (dest / "error.log").write_text(errors[key], encoding="utf-8")
        for q in questions:
            for group, views in GROUPS.items():
                dest = out / q["id"] / group
                dest.mkdir(parents=True)
                started = time.perf_counter()
                row = dict(question_id=q["id"], window=q["window"], group=group,
                           reference_answer=q["reference_answer"], status="running",
                           verdict="pending", component_scores={}, review_notes="")
                rows.append(row)
                write_json(dest / "question.json", q)
                write_json(dest / "metrics.json", {"input_tokens": None, "generate_seconds": None})
                (dest / "error.log").write_text("", encoding="utf-8")
                prompt = prompt_for(q)
                (dest / "prompt.txt").write_text(prompt, encoding="utf-8")
                try:
                    missing = [f"{q['window']}_{v}" for v in views if f"{q['window']}_{v}" in errors]
                    if missing:
                        raise RuntimeError("\n".join(errors[k] for k in missing))
                    records = [f for v in views for f in sampled[f"{q['window']}_{v}"]]
                    write_json(dest / "frames.json", records)
                    if args.extract_only:
                        row["status"] = "extracted"
                    else:
                        if runtime is None and load_error is None:
                            try:
                                runtime = runtime_factory(args.model)
                                meta["runtime"] = runtime.metadata
                            except Exception:
                                load_error = traceback.format_exc()
                        if load_error:
                            raise RuntimeError(load_error)
                        raw, metrics = runtime(records, args.max_new_tokens, dest, prompt=prompt)
                        (dest / "raw_response.txt").write_text(raw, encoding="utf-8")
                        write_json(dest / "metrics.json", metrics)
                        row.update(status="completed", raw_response=raw, metrics=metrics)
                except Exception:
                    row["status"] = "failed"
                    (dest / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
                finally:
                    row["total_seconds"] = time.perf_counter() - started
                    write_json(dest / "run.json", row)
                    write_json(out / "review.json", {"status": "pending_human_semantic_review", "rows": rows})
                    print(f"{q['id']} / {group}: {row['status']}", flush=True)
        meta["status"] = "failed" if any(r["status"] == "failed" for r in rows) else (
            "extracted" if args.extract_only else "completed_pending_review")
    except BaseException:
        meta["status"] = "interrupted_or_failed"
        (out / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        meta["tasks"] = [{k: r[k] for k in ("question_id", "group", "status")} for r in rows]
        write_json(out / "run.json", meta)
    return 1 if meta["status"] == "failed" else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/phase1_qa_v1.json"))
    p.add_argument("--clips", type=Path, required=True)
    p.add_argument("--question-ids", nargs="+")
    p.add_argument("--model", type=Path)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--extract-only", action="store_true")
    args = p.parse_args()
    if args.frames < 1 or args.max_new_tokens < 1:
        p.error("Frame and token limits must be positive")
    if not args.extract_only and args.model is None:
        p.error("--model required for inference")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
