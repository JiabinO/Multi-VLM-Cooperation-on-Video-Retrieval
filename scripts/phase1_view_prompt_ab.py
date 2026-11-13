"""Paired test of event observation with and without an approved camera-view profile."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

from video_smoke import write_json
from w02_qa import VIEWS, QwenRuntime, extract, git_info

WINDOWS = {"W01": 20, "W02": 60, "W03": 100, "W04": 140, "W05": 180, "W06": 220}
FACT_CATEGORIES = {"object", "action", "state_change", "hand_activity", "occlusion"}
BASE_PROMPT = '''任务：从本次输入的同一机位视频抽样帧中，提取与当前窗口有关的内容，重点观察物体、操作者动作、可见状态变化、双手活动以及遮挡；不要重复泛泛描述背景。
只报告本次输入画面直接支持的内容。不要仅凭常识、机位历史描述或未采样时段补全事实，不把先后相邻帧自动解释为连续动作。画面中没有显示不等于现实中不存在。
将可直接观察到的事实与仍无法确认的细节分开。每条可见事实要给出原子化描述、支持帧编号、具体视觉证据和 0 到 1 的模型自评置信度。每条不确定项要给出相关帧编号和无法判断的原因。没有可靠事实时可用空列表。仅输出 JSON：
{"visible_facts":[{"category":"object","claim":"可见事实","frame_ids":[0],"evidence":"帧中具体可见内容","confidence":0.8}],"uncertain_items":[{"detail":"仍无法确认的细节","frame_ids":[0],"reason":"遮挡、采样不足或图像证据不足的原因"}]}
category 只可为 object、action、state_change、hand_activity、occlusion。frame_ids 必须引用当前窗口输入帧编号。置信度是模型自评，不是校准概率。不要生成问题、参考答案或评分。所有内容待人工审核。'''
PROFILE_BLOCK = '''\n\n以下是人工审核通过的机位视野描述，只是历史视野先验，帮助定位画面区域；不代表当前窗口一定出现相关物体或事件。当前输入图像优先：每条事实仍须由当前帧直接支持；若先验与图像不符，以图像为准；不得把先验本身写成观察事实。\n机位视野描述：{profile}'''


def load_profiles(path, views):
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles = data.get("profiles") if isinstance(data, dict) else None
    if not isinstance(profiles, dict):
        raise ValueError("Profiles file must contain a profiles object")
    selected = {}
    for view in views:
        item = profiles.get(view)
        if not isinstance(item, dict) or item.get("approved") is not True:
            raise ValueError(f"{view} must have approved=true after human review")
        text = item.get("view_coverage")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{view} needs nonempty view_coverage")
        selected[view] = text.strip()
    return selected


def parse_observation(raw, records):
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")
    for key in ("visible_facts", "uncertain_items"):
        if not isinstance(data.get(key), list):
            raise ValueError(f"Expected {key} list")
    valid_ids = {r["frame_id"] for r in records}
    for fact in data["visible_facts"]:
        if not isinstance(fact, dict) or fact.get("category") not in FACT_CATEGORIES:
            raise ValueError("Invalid visible fact category")
        if not isinstance(fact.get("claim"), str) or not fact["claim"].strip():
            raise ValueError("Visible fact needs a claim")
        ids = fact.get("frame_ids")
        if not isinstance(ids, list) or not ids or any(type(i) is not int or i not in valid_ids for i in ids):
            raise ValueError("Visible fact has invalid frame references")
        if not isinstance(fact.get("evidence"), str) or not fact["evidence"].strip():
            raise ValueError("Visible fact needs visual evidence")
        conf = fact.get("confidence")
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
            raise ValueError("Confidence must be between 0 and 1")
    for item in data["uncertain_items"]:
        if not isinstance(item, dict) or not isinstance(item.get("detail"), str) or not item["detail"].strip():
            raise ValueError("Uncertain item needs detail")
        ids = item.get("frame_ids")
        if not isinstance(ids, list) or not ids or any(type(i) is not int or i not in valid_ids for i in ids):
            raise ValueError("Uncertain item has invalid frame references")
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ValueError("Uncertain item needs a reason")
    return data


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_review(out, pairs):
    review = {"status": "pending_human_review", "is_gold": False,
              "conditions": ["no_profile", "approved_profile"], "pairs": pairs,
              "instructions": "逐对比较相同窗口、相同抽样帧下两种提示的可见事实和不确定项。检查新增事实是否有输入帧支持、是否遗漏、是否把先验照抄成当前观察。记录事实覆盖、无依据新增、错误、uncertainty 处理和人工审核；置信度仅作模型自评。"}
    write_json(out / "review.json", review)
    lines = ["# 机位先验提示配对实验审核表", "",
             "比较同一机位、同一窗口、同一帧的 no_profile 与 approved_profile 输出。",
             "重点检查提示是否减少遗漏，或引入无画面证据的先验复述。", ""]
    for pair in pairs:
        key = pair["pair_id"]
        lines += [f"## {key} — {pair['status']}", "",
                  f"[采样预览]({key}/samples/preview.jpg) · [帧清单]({key}/frames.json)",
                  f"窗口：{pair['window']}；机位：{pair['view']}；输入帧完全相同：是", ""]
        for condition in ("no_profile", "approved_profile"):
            result = pair["conditions"].get(condition, {})
            lines += [f"### {condition} — {result.get('status', 'missing')}",
                      f"[原始回答]({key}/{condition}/raw_response.txt) · [prompt]({key}/{condition}/prompt.txt) · [错误日志]({key}/{condition}/error.log)", ""]
            for fact in result.get("visible_facts", []):
                lines += [f"- 可见事实（{fact['category']}）：{fact['claim']}；支持帧 {fact['frame_ids']}；自评置信度 {fact['confidence']}；证据：{fact['evidence']}"]
            for item in result.get("uncertain_items", []):
                lines += [f"- 不确定：{item['detail']}；相关帧 {item['frame_ids']}；原因：{item['reason']}"]
            lines.append("")
        lines += ["### 人工配对判断", "- 机位提示组减少的遗漏：", "- 机位提示组新增但无画面证据的内容：",
                  "- 两组共同错误或共同遗漏：", "- 不确定性表达变化：", "- 总体审核结论：", ""]
    (out / "review.md").write_text("\n".join(lines), encoding="utf-8")


def run(args, extractor=extract, runtime_factory=QwenRuntime):
    out = args.output
    profiles = None
    if not args.extract_only:
        if args.profiles is None:
            raise ValueError("--profiles is required for inference; do not use unreviewed raw responses as prompts")
        profiles = load_profiles(args.profiles, args.views)
    out.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    meta = dict(status="running", experiment_id="Assembly101-view-profile-prompt-ab",
                prompt_version="view-prompt-ab-v1", conditions=["no_profile", "approved_profile"],
                config=config, command=sys.argv, python=sys.version,
                started_at=datetime.now(timezone.utc).isoformat(), **git_info())
    if args.profiles and args.profiles.is_file():
        meta["profiles_sha256"] = sha256(args.profiles)
        if profiles is not None:
            write_json(out / "profiles.snapshot.json", {v: profiles[v] for v in args.views})
    write_json(out / "run.json", meta)
    (out / "error.log").write_text("", encoding="utf-8")
    pairs, prepared, runtime, load_error = [], [], None, None
    try:
        for index, (window, view) in enumerate((w, v) for w in args.windows for v in args.views):
            key = f"{window}_{view}"
            pair_out = out / key
            pair = dict(pair_id=key, window=window, view=view, status="sampling",
                        conditions={}, paired_input_sha256=None, execution_order=[], is_gold=False)
            pairs.append(pair)
            try:
                video = args.clips / f"{key}.mp4"
                if not video.is_file():
                    raise FileNotFoundError(video)
                data = extractor(video, pair_out / "samples", args.frames, clip_start=WINDOWS[window])
                records = [dict(frame,
                    absolute_file=str((pair_out / "samples" / frame["file"]).resolve()),
                    label=f"Window {window}, frame {frame['frame_id']}, time={frame['nominal_time_sec']:.3f}s")
                    for frame in data["frames"]]
                if not records:
                    raise ValueError("No sampled frames")
                write_json(pair_out / "frames.json", records)
                pair["paired_input_sha256"] = [f["sha256"] for f in records]
                pair["sampled_frames"] = [{k: f[k] for k in ("frame_id", "frame_index", "nominal_time_sec", "sha256") if k in f} for f in records]
                pair["status"] = "sampled"
                prepared.append((index, pair, pair_out, records))
            except Exception:
                pair["status"] = "extraction_failed"
                pair_out.mkdir(parents=True, exist_ok=True)
                (pair_out / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
                print(f"{key}: extraction_failed", flush=True)

        if args.extract_only:
            for _, pair, pair_out, _ in prepared:
                pair["status"] = "extracted"
                for condition in ("no_profile", "approved_profile"):
                    d = pair_out / condition
                    d.mkdir(parents=True, exist_ok=True)
                    (d / "error.log").write_text("", encoding="utf-8")
                    pair["conditions"][condition] = {"status": "extracted"}
        elif prepared:
            try:
                runtime = runtime_factory(args.model)
                meta["runtime"] = runtime.metadata
            except Exception:
                load_error = traceback.format_exc()
            for index, pair, pair_out, records in prepared:
                conditions = ["no_profile", "approved_profile"]
                if index % 2:
                    conditions.reverse()
                pair["execution_order"] = conditions
                for condition in conditions:
                    condition_out = pair_out / condition
                    condition_out.mkdir(parents=True, exist_ok=True)
                    prompt = BASE_PROMPT
                    if condition == "approved_profile":
                        prompt += PROFILE_BLOCK.format(profile=profiles[pair["view"]])
                    (condition_out / "prompt.txt").write_text(prompt, encoding="utf-8")
                    write_json(condition_out / "frames.json", records)
                    (condition_out / "error.log").write_text("", encoding="utf-8")
                    result = {"status": "running", "condition": condition,
                              "window": pair["window"], "view": pair["view"],
                              "profile_text": profiles[pair["view"]] if condition == "approved_profile" else None}
                    write_json(condition_out / "run.json", result)
                    started = time.perf_counter()
                    try:
                        if load_error:
                            raise RuntimeError(load_error)
                        raw, metrics = runtime(records, args.max_new_tokens, condition_out, prompt=prompt)
                        (condition_out / "raw_response.txt").write_text(raw, encoding="utf-8")
                        write_json(condition_out / "metrics.json", metrics)
                        result["status"] = "parse_failed"
                        parsed = parse_observation(raw, records)
                        write_json(condition_out / "parsed.json", parsed)
                        result.update(parsed)
                        result["status"] = "completed"
                    except Exception:
                        if result["status"] != "parse_failed":
                            result["status"] = "failed"
                        (condition_out / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
                    finally:
                        result["total_seconds"] = time.perf_counter() - started
                        write_json(condition_out / "run.json", result)
                        pair["conditions"][condition] = result
                pair["status"] = "completed" if all(pair["conditions"].get(c, {}).get("status") == "completed"
                    for c in ("no_profile", "approved_profile")) else "partial_or_failed"
                print(f"{pair['pair_id']}: {pair['status']}", flush=True)
        meta["status"] = "extracted" if args.extract_only and all(p["status"] == "extracted" for p in pairs) else (
            "completed_pending_review" if not args.extract_only and pairs and all(p["status"] == "completed" for p in pairs)
            else "failed")
    except BaseException:
        meta["status"] = "interrupted_or_failed"
        (out / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        meta["pairs"] = [{"pair_id": p["pair_id"], "status": p["status"],
                          "conditions": {k: v.get("status") for k, v in p["conditions"].items()}}
                         for p in pairs]
        write_review(out, pairs)
        write_json(out / "run.json", meta)
    return 1 if meta["status"] == "failed" else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clips", required=True, type=Path)
    p.add_argument("--profiles", type=Path, help="Human-approved camera view profiles JSON")
    p.add_argument("--windows", nargs="+", choices=list(WINDOWS), default=["W03", "W04", "W05", "W06"])
    p.add_argument("--views", nargs="+", choices=list(VIEWS), default=list(VIEWS))
    p.add_argument("--model", type=Path)
    p.add_argument("--frames", type=int, default=8, help="Frames per 5-second clip, shared by both conditions")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--extract-only", action="store_true")
    args = p.parse_args()
    if args.frames < 1 or args.max_new_tokens < 1:
        p.error("Frame and token counts must be positive")
    if len(set(args.windows)) != len(args.windows) or len(set(args.views)) != len(args.views):
        p.error("Duplicate windows or views")
    if not args.extract_only and args.model is None:
        p.error("--model required for inference")
    try:
        return run(args)
    except ValueError as exc:
        p.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
