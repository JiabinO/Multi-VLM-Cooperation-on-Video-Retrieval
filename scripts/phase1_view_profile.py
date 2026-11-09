"""Draft per-camera scene and visibility profiles from sampled clips for human review."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

from video_smoke import write_json
from w02_qa import VIEWS, QwenRuntime, extract, git_info

WINDOWS = {"W01": 20, "W02": 60, "W03": 100, "W04": 140, "W05": 180, "W06": 220}
CATEGORIES = {"scene_layout", "view_coverage", "recurring_occlusion", "temporal_content", "uncertainty"}
PROMPT = '''你正在根据同一摄像机在多个时间窗的稀疏抽样帧，起草“机位视角描述”，供人工审核；这些帧不是连续视频，也不是事实真值。
只使用当前摄像机输入的图像，不借用其他视角，不根据常识补全画外内容。按类别逐条写原子观察：
- scene_layout：多个时间窗中反复出现的稳定场景布局或背景特征；只有重复得到图像支持时才称为稳定。
- view_coverage：这个画面直接覆盖到的区域及可见边界，例如桌面、操作者、模型各自出现在哪里；不要推断摄像机的物理位置或朝向。
- recurring_occlusion：反复出现的遮挡关系，以及因此看不清的内容。
- temporal_content：某些窗口/帧中出现的临时物体、手部或动作，只描述所给帧，不写成机位固有特征。
- uncertainty：采样不足、模糊、遮挡或仅凭图像无法判断的内容。
不要声称某信息是“其他视角看不到”或“本视角独有”，因为你没有看到其他摄像机输入。不得将“画面未显示”写成“场景中不存在”。
每条观察都必须给出明确的视觉证据、引用输入帧编号和 0 到 1 的自评置信度；无可靠观察时可返回空列表。仅输出 JSON：
{"observations":[{"category":"scene_layout","claim":"一个原子观察","frame_ids":[0,1],"evidence":"这些编号帧中实际可见的证据","confidence":0.8,"uncertainty":"仍不能确定的部分；没有则为空字符串"}]}
frame_ids 引用本摄像机所有输入帧共用的编号。置信度不是校准准确率。不要生成 QA、参考答案或评分标准。所有内容待人工审核。'''


def parse_profile(raw, records):
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("observations"), list):
        raise ValueError("Expected observations list")
    valid_ids = {f["profile_frame_id"] for f in records}
    for obs in data["observations"]:
        if not isinstance(obs, dict) or obs.get("category") not in CATEGORIES:
            raise ValueError("Invalid profile category")
        if not isinstance(obs.get("claim"), str) or not obs["claim"].strip():
            raise ValueError("Expected nonempty claim")
        ids = obs.get("frame_ids")
        if not isinstance(ids, list) or not ids or any(
                type(i) is not int or i not in valid_ids for i in ids):
            raise ValueError("Invalid frame reference")
        if not isinstance(obs.get("evidence"), str) or not obs["evidence"].strip():
            raise ValueError("Expected nonempty visual evidence")
        confidence = obs.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be a number between 0 and 1")
        if not isinstance(obs.get("uncertainty"), str):
            raise ValueError("Expected uncertainty string")
    return data


def write_review(out, tasks):
    write_json(out / "review.json", {
        "status": "pending_human_review", "is_gold": False,
        "decisions": ["pending", "confirm", "correct", "delete", "uncertain"],
        "instructions": "审核稳定布局、画面覆盖范围、反复遮挡和窗口内临时内容；检查每条视觉证据及支持帧。模型未看其他机位，不能自行判定共享或独有信息；需跨视角比对后再归类。模型自评置信度不代表校准准确率。",
        "tasks": tasks})
    lines = ["# 机位视角描述人工审核表：模型草稿，不是真值", "",
             "每条选择 pending / confirm / correct / delete / uncertain。检查证据帧。",
             "稳定背景需在多个窗口重复出现；单个窗口中的手、零件和动作属于临时内容。",
             "模型只看当前机位，不能据此认定信息在其他机位不可见。跨视角比对后再标共享、独有或更清楚。", ""]
    for task in tasks:
        key = task["task_id"]
        previews = " · ".join(
            f"[{window}预览]({key}/samples/{window}/preview.jpg)" for window in task["sampled_windows"])
        lines += [f"## {key} — {task['status']}", "",
                  f"{previews} · [原始回答]({key}/raw_response.txt) · [错误日志]({key}/error.log)",
                  f"采样窗口：{', '.join(task['windows'])}；每个窗口 {task['frames_per_window']} 帧。", ""]
        for obs in task["observations"]:
            lines += [f"### {obs['id']} / {obs['category']}", f"模型草稿：{obs['claim']}",
                      f"支持帧：{obs['frame_ids']}",
                      f"模型置信度（自评，非准确率）：{obs['confidence']}",
                      f"模型描述的视觉证据：{obs['evidence']}",
                      f"模型不确定处：{obs['uncertainty']}",
                      "- 判定：pending", "- 修正后事实：", "- 审核备注：", ""]
        lines += ["### 人工补充", "- 多窗口重复出现的稳定场景信息：", "- 本机位画面覆盖范围：",
                  "- 反复出现的遮挡/盲区：", "- 单窗口临时内容：", "- 与其他机位比较后的共享信息：",
                  "- 与其他机位比较后的独有或更清楚信息：", "- 尚不能确定的信息：", ""]
    (out / "review.md").write_text("\n".join(lines), encoding="utf-8")


def run(args, extractor=extract, runtime_factory=QwenRuntime):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    meta = dict(status="running", experiment_id="Assembly101-camera-view-profile",
                prompt_version="camera-view-profile-v1", config=config, command=sys.argv,
                python=sys.version, started_at=datetime.now(timezone.utc).isoformat(), **git_info())
    write_json(out / "run.json", meta)
    (out / "error.log").write_text("", encoding="utf-8")
    tasks, runtime, load_error = [], None, None
    try:
        prepared = {}
        for view in VIEWS:
            key, dest = view, out / view
            task = dict(task_id=key, view=view, windows=list(args.windows),
                        frames_per_window=args.frames_per_window, status="sampling",
                        observations=[], sampling_errors={}, human_additions={},
                        cross_view_comparison="pending", is_gold=False)
            tasks.append(task)
            records = []
            frame_id = 0
            started = time.perf_counter()
            try:
                for window in args.windows:
                    video = args.clips / f"{window}_{view}.mp4"
                    window_out = dest / "samples" / window
                    try:
                        if not video.is_file():
                            raise FileNotFoundError(video)
                        data = extractor(video, window_out, args.frames_per_window,
                                         clip_start=WINDOWS[window])
                        for frame in data["frames"]:
                            records.append(dict(frame, profile_frame_id=frame_id, window=window,
                                absolute_file=str((window_out / frame["file"]).resolve()),
                                label=f"Window {window}; local frame {frame['frame_id']}"))
                            frame_id += 1
                    except Exception:
                        task["sampling_errors"][window] = traceback.format_exc()
                task["sampled_frame_count"] = len(records)
                task["sampled_windows"] = sorted({r["window"] for r in records})
                if not records:
                    raise RuntimeError("No frames sampled for this camera")
                write_json(dest / "frames.json", records)
                (dest / "prompt.txt").write_text(PROMPT, encoding="utf-8")
                if args.extract_only:
                    task["status"] = "extracted"
                else:
                    prepared[view] = (task, dest, records)
            except Exception:
                task["status"] = "failed"
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            finally:
                dest.mkdir(parents=True, exist_ok=True)
                if task["sampling_errors"]:
                    error_text = "\n\n".join(
                        f"[{window}]\n{error}" for window, error in task["sampling_errors"].items())
                    (dest / "error.log").write_text(error_text, encoding="utf-8")
                elif not (dest / "error.log").exists():
                    (dest / "error.log").write_text("", encoding="utf-8")
                task["sampling_seconds"] = time.perf_counter() - started
                write_json(dest / "task.json", task)
                print(f"{view}: {task['status']}", flush=True)

        if not args.extract_only and prepared:
            try:
                runtime = runtime_factory(args.model)
                meta["runtime"] = runtime.metadata
            except Exception:
                load_error = traceback.format_exc()
        for view, (task, dest, records) in prepared.items():
            started = time.perf_counter()
            try:
                if load_error:
                    raise RuntimeError(load_error)
                (dest / "prompt.txt").write_text(PROMPT, encoding="utf-8")
                raw, metrics = runtime(records, args.max_new_tokens, dest, prompt=PROMPT)
                (dest / "raw_response.txt").write_text(raw, encoding="utf-8")
                write_json(dest / "metrics.json", metrics)
                task["status"] = "parse_failed"
                draft = parse_profile(raw, records)
                write_json(dest / "draft.json", draft)
                task["observations"] = [dict(obs, id=f"{view}-{i:03d}", decision="pending",
                    corrected_claim="", review_notes="") for i, obs in enumerate(draft["observations"])]
                task["status"] = "drafted"
            except Exception:
                if task["status"] != "parse_failed":
                    task["status"] = "failed"
                (dest / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            finally:
                task["inference_seconds"] = time.perf_counter() - started
                write_json(dest / "task.json", task)
                print(f"{view}: {task['status']}", flush=True)
        failures = any(t["status"] in {"failed", "parse_failed"} for t in tasks)
        meta["status"] = "failed" if failures else ("extracted" if args.extract_only else "drafted_pending_review")
    except BaseException:
        meta["status"] = "interrupted_or_failed"
        (out / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        write_review(out, tasks)
        meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        meta["tasks"] = [{k: t[k] for k in ("task_id", "status", "sampled_frame_count") if k in t}
                         for t in tasks]
        write_json(out / "run.json", meta)
    return 1 if meta["status"] == "failed" else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clips", type=Path, required=True, help="Directory of pretrimmed Wxx_camera.mp4 clips")
    p.add_argument("--windows", nargs="+", choices=list(WINDOWS), default=["W03", "W04", "W05", "W06"])
    p.add_argument("--model", type=Path)
    p.add_argument("--frames-per-window", type=int, default=2,
                   help="Frames sampled per window and camera; defaults to 2 to compare stable versus transient content")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--extract-only", action="store_true")
    args = p.parse_args()
    if args.frames_per_window < 1 or args.max_new_tokens < 1:
        p.error("Frame and token limits must be positive")
    if len(set(args.windows)) != len(args.windows):
        p.error("Duplicate windows")
    if not args.extract_only and args.model is None:
        p.error("--model required for inference")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
