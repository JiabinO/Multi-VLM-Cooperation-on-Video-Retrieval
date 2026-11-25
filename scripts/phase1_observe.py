"""Generate per-view observation drafts and human review forms, not gold QA."""
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
CATEGORIES = {"object", "action", "state_change", "left_hand", "right_hand", "occlusion"}
PROMPT = '''你正在帮助人工整理视频事实草稿，不是在回答问题或生成真值。
只描述当前这一摄像机采样图中可见的事实，不借用其他视角，不补全未观察到的动作。
关注被操作的组件、动作、前后可见状态、左右手的作用、遮挡与看不清的细节。
不确定组件名称时描述外观和位置，不强行命名。不确定拧紧或拧松时只写拧动。
一条只写一个事实。不要把单帧姿态当成已证实的完整动作，不推断动机。
用中文，仅输出 JSON，格式：
{"observations":[{"category":"action","claim":"可见事实","frame_ids":[0,1],"evidence":"第0帧中右手握着工具并接触红色螺丝；第1帧仍可见接触","confidence":0.82,"uncertainty":"螺丝是否已拆下无法从这些帧确认"}]}
category 只可为 object、action、state_change、left_hand、right_hand、occlusion。
frame_ids 引用输入帧编号，仅用于人工核查，不作为理解能力评分。evidence 必须具体描述这些帧里支持 claim 的可见内容；不要写未出现在图像中的推断。confidence 为 0 到 1 的数字，表示你对这条 claim 的主观把握，不是正确率保证；证据不足时应降低置信度并说明原因。uncertainty 写尚不能确定的内容；没有则写空字符串。没有可靠观察时输出空列表。
不要生成问题、参考答案或评分。所有内容均待人工确认。'''


def parse_draft(raw, records):
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("observations"), list):
        raise ValueError("Expected observations list")
    valid_ids = {f["frame_id"] for f in records}
    for obs in data["observations"]:
        if not isinstance(obs, dict) or obs.get("category") not in CATEGORIES:
            raise ValueError("Invalid observation category")
        if not isinstance(obs.get("claim"), str) or not obs["claim"].strip():
            raise ValueError("Expected nonempty claim")
        ids = obs.get("frame_ids")
        if not isinstance(ids, list) or not ids or any(type(i) is not int or i not in valid_ids for i in ids):
            raise ValueError("Invalid frame reference")
        if not isinstance(obs.get("evidence"), str) or not obs["evidence"].strip():
            raise ValueError("Expected nonempty visual evidence description")
        confidence = obs.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be a number between 0 and 1")
        if not isinstance(obs.get("uncertainty"), str):
            raise ValueError("Expected uncertainty string")
    return data


def write_review(out, tasks):
    review = {"status": "pending_human_review", "is_gold": False,
              "decisions": ["pending", "confirm", "correct", "delete", "uncertain"],
              "instructions": "审核对象与动作、松紧方向、前后状态、左右手及遮挡；逐条核查置信度和视觉证据是否有帧支持，并补充模型遗漏。模型置信度是自评，不代表校准准确率；帧引用仅用于核查。",
              "tasks": tasks}
    write_json(out / "review.json", review)
    lines = ["# 人工审核表：模型草稿，不是真值", "",
             "每条选择 pending / confirm / correct / delete / uncertain；修正时填写正确事实。",
             "必须补充遗漏，尤其是本视角独有的信息；看不清不能写成没有发生。",
             "审核结构不自动转成 QA；后续只用确认或修正后的事实出题。", ""]
    for task in tasks:
        key = task["task_id"]
        lines += [f"## {key} — {task['status']}", "",
                  f"[预览]({key}/preview.jpg) · [原始回答]({key}/raw_response.txt) · [错误日志]({key}/error.log)", ""]
        if task.get("reached_token_limit"):
            lines += ["注意：生成达到 token 上限，草稿可能不完整，需核查原始回答。", ""]
        for obs in task["observations"]:
            lines += [f"### {obs['id']} / {obs['category']}",
                      f"模型草稿：{obs['claim']}", f"支持帧：{obs['frame_ids']}",
                      f"模型置信度（自评，非准确率）：{obs['confidence']}",
                      f"模型描述的视觉证据：{obs['evidence']}",
                      f"模型不确定处：{obs['uncertainty']}",
                      "- 判定：pending", "- 修正后事实：", "- 审核备注：", ""]
        lines += ["### 每路都需填写（即使模型没有输出事实）", "- 模型遗漏的事实：",
                  "- 该视角独有或更清楚的信息：", "- 看不清、不能确认的细节：",
                  "- 是否看过完整片段，而非仅预览图：否", ""]
    (out / "review.md").write_text("\n".join(lines), encoding="utf-8")


def run(args, extractor=extract, runtime_factory=QwenRuntime):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    meta = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(),
                command=sys.argv, python=sys.version, **git_info(),
                config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    tasks = []
    runtime = None
    load_error = None
    write_json(out / "run.json", meta)
    (out / "error.log").write_text("", encoding="utf-8")
    try:
        for window in args.windows:
            for view in VIEWS:
                key = f"{window}_{view}"
                dest = out / key
                video = args.clips / f"{key}.mp4"
                task = dict(task_id=key, window=window, view=view, source_video=str(video.resolve()),
                            source_offset_sec=WINDOWS[window], status="running", observations=[],
                            human_additions=[], unique_information="", unclear_details="",
                            full_clip_reviewed=False)
                tasks.append(task)
                started = time.perf_counter()
                try:
                    if not video.is_file():
                        raise FileNotFoundError(video)
                    data = extractor(video, dest, args.frames, clip_start=WINDOWS[window])
                    records = [dict(f, absolute_file=str((dest / f["file"]).resolve()),
                        label=f"View {view}, frame {f['frame_id']}, source nominal time={f['nominal_time_sec']:.3f}s")
                        for f in data["frames"]]
                    if not records:
                        raise ValueError("No sampled frames")
                    (dest / "prompt.txt").write_text(PROMPT, encoding="utf-8")
                    if args.extract_only:
                        task["status"] = "extracted"
                    else:
                        if runtime is None and load_error is None:
                            try:
                                runtime = runtime_factory(args.model)
                                meta["runtime"] = runtime.metadata
                            except Exception:
                                load_error = traceback.format_exc()
                        if load_error:
                            raise RuntimeError(load_error)
                        raw, metrics = runtime(records, args.max_new_tokens, dest, prompt=PROMPT)
                        (dest / "raw_response.txt").write_text(raw, encoding="utf-8")
                        write_json(dest / "metrics.json", metrics)
                        task["reached_token_limit"] = metrics.get("reached_token_limit")
                        task["status"] = "parse_failed"
                        draft = parse_draft(raw, records)
                        write_json(dest / "draft.json", draft)
                        task["observations"] = [dict(o, id=f"{key}-{i:03d}", decision="pending",
                            corrected_claim="", review_notes="") for i, o in enumerate(draft["observations"])]
                        task["status"] = "drafted"
                except Exception:
                    if task["status"] != "parse_failed":
                        task["status"] = "failed"
                    dest.mkdir(parents=True, exist_ok=True)
                    (dest / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
                finally:
                    dest.mkdir(parents=True, exist_ok=True)
                    if not (dest / "error.log").exists():
                        (dest / "error.log").write_text("", encoding="utf-8")
                    task["total_seconds"] = time.perf_counter() - started
                    write_json(dest / "task.json", task)
                    write_review(out, tasks)
                    print(f"{key}: {task['status']}", flush=True)
        meta["status"] = "failed" if any(t["status"] in {"failed", "parse_failed"} for t in tasks) else (
            "extracted" if args.extract_only else "drafted_pending_review")
    except BaseException:
        meta["status"] = "interrupted_or_failed"
        (out / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        meta["tasks"] = [{k: t[k] for k in ("task_id", "status", "total_seconds") if k in t} for t in tasks]
        meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "run.json", meta)
    return 1 if meta["status"] == "failed" else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clips", type=Path, required=True, help="Directory of pretrimmed 5s Wxx_camera.mp4 clips")
    p.add_argument("--windows", nargs="+", choices=list(WINDOWS), default=["W03", "W04", "W05", "W06"])
    p.add_argument("--model", type=Path)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--extract-only", action="store_true")
    args = p.parse_args()
    if args.frames < 1 or args.max_new_tokens < 1:
        p.error("Frame and token limits must be positive")
    if len(set(args.windows)) != len(args.windows):
        p.error("Duplicate windows")
    if not args.extract_only and args.model is None:
        p.error("--model required for inference")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
