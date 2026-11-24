"""Diagnose a query-conditioned, per-view evidence graph on one MVVBench query.

The default case is the monitor-count question 63am_lobby_3-q01190. This is a
small diagnostic, not a benchmark result: it separates per-view observations,
cross-view entity deduplication, and final arithmetic while preserving every
raw model response and frame reference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any

from diagnose_mvvbench_evidence import annotation_index, mapped_video_path, sample_timestamp_frames
from video_evidence import answer_options, get_runtime, load_mapping, question_id, selected_views


DEFAULT_QUERY_ID = "63am_lobby_3-q01190"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def choice_options(question: dict[str, Any], annotation: dict[str, Any]) -> dict[str, str]:
    options = question.get("options")
    if isinstance(options, dict) and all(isinstance(options.get(k), str) for k in "ABCD"):
        return {k: options[k] for k in "ABCD"}
    return answer_options(annotation)


def answer_letter(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.match(r"^\s*([ABCD])(?:\b|[.:)\s])", value, re.IGNORECASE)
    return match.group(1).upper() if match else None


def answer_prompt(question: str, options: dict[str, str], labels: list[str]) -> str:
    return (
        "根据给定图像回答选择题。先核对题目指定的计数对象和属性，再计算选项所问的差值。"
        "只依据可见内容；模糊或遮挡的对象不要武断计数。只输出 JSON："
        "{\"answer\":\"A|B|C|D\",\"confidence\":\"low|medium|high\",\"brief_evidence\":\"简短可核对依据\"}。\n"
        + "\n".join(labels) + "\n问题：" + question + "\n选项：\n"
        + "\n".join(f"{k}. {v}" for k, v in options.items())
    )


def graph_extraction_prompt(question: str, view_id: str, frame_info: list[dict[str, Any]]) -> str:
    frame_lines = "\n".join(
        f"frame_id={i}; timestamp={x['time_s']:.3f}s" for i, x in enumerate(frame_info)
    )
    return (
        "你是一个视角端的视觉观察器。仅分析当前单个摄像机视角，服务于下面的问题；"
        "本阶段不回答选择题，不看候选答案，不推测其他视角。\n"
        "请建立简洁的局部证据图：entities记录每个可辨认的显示器，包括笔记本电脑屏幕；"
        "同一显示器在多帧反复出现时尽量复用同一个local_id。每个实体记录类别、外观、画面位置、"
        "屏幕状态（appears_on / appears_off / uncertain / not_visible）、可见性（clear / partial / occluded）、"
        "以及哪些帧支持该观察。只有屏幕内容/亮屏特征直接可见时才判断appears_on；黑色外观本身不足以断定关机。"
        "看不到或被遮挡的显示器不要补造。relations只记录可见的相对位置。claims须引用实体ID和帧号。"
        "不确定时保留 uncertain，并在limitations说明。只输出 JSON，结构为："
        "{\"view_id\":\"...\",\"entities\":[{\"local_id\":\"...\",\"category\":\"monitor|laptop_screen|other\","
        "\"appearance\":\"...\",\"position\":\"...\",\"screen_state\":\"appears_on|appears_off|uncertain|not_visible\","
        "\"state_confidence\":\"low|medium|high\",\"visibility\":\"clear|partial|occluded\","
        "\"frame_ids\":[0],\"evidence\":\"直接可见依据\"}],"
        "\"relations\":[{\"subject\":\"local_id\",\"predicate\":\"left_of|right_of|near|behind\",\"object\":\"local_id\",\"frame_ids\":[0]}],"
        "\"claims\":[{\"claim\":\"...\",\"entity_ids\":[\"...\"],\"frame_ids\":[0],\"support\":\"direct|uncertain\"}],"
        "\"limitations\":[\"...\"]}.\n"
        f"当前视角：View {view_id}\n本次输入帧：\n{frame_lines}\n问题：{question}"
    )


def graph_fusion_prompt(question: str, options: dict[str, str], graphs: list[dict[str, Any]],
                        deduplicate: bool) -> str:
    if deduplicate:
        instruction = (
            "中心端任务：先审查不同视角的局部实体，再判断哪些记录可能对应同一台物理显示器。"
            "不能仅凭类别或相似外观合并；要结合位置关系、外观、周围实体和来源帧。"
            "匹配状态只能是 same / different / uncertain，并给出简短依据。只有 same 才去重；uncertain 要保留为不确定候选，"
            "同时报告去重与不确定性对计数的影响。之后分别统计 appears_on 与 appears_off 的实体，"
            "按题意计算两者数量差的绝对值，并映射到选项。若屏幕状态 uncertain，不要强行归入开或关。"
        )
    else:
        instruction = (
            "中心端消融条件：禁止跨视角匹配或去重，把各视角端侧记录按记录数直接汇总。"
            "之后统计 appears_on 与 appears_off 并计算数量差的绝对值。这个条件用于展示重复计数风险，"
            "不能把它解释为合理的部署策略。"
        )
    schema = (
        "只输出 JSON：{\"cross_view_matches\":[{\"entities\":[{\"view_id\":\"...\",\"local_id\":\"...\"}],"
        "\"status\":\"same|different|uncertain\",\"basis\":\"...\"}],"
        "\"count_on\":0,\"count_off\":0,\"count_uncertain\":0,\"difference\":0,"
        "\"calculation\":\"...\",\"answer\":\"A|B|C|D|unknown\",\"confidence\":\"low|medium|high\","
        "\"limitations\":[\"...\"]}。"
    )
    return (
        "你是多视角中心融合器。输入是不同摄像机各自生成的查询相关局部证据图。"
        "局部描述是模型观察，不保证正确；不得把它们当作标注真值。只使用输入证据，不补充未观察到的实体。\n"
        + instruction + "\n" + schema + "\n问题：" + question + "\n选项：\n"
        + "\n".join(f"{k}. {v}" for k, v in options.items())
        + "\n各视角证据图：\n" + json.dumps(graphs, ensure_ascii=False, indent=2)
    )


def run(args: argparse.Namespace) -> int:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    ds = config["dataset"]
    annotations = annotation_index(Path(ds["annotation_path"]))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if isinstance(questions, dict):
        questions = questions.get("data", questions.get("questions"))
    if not isinstance(questions, list):
        raise ValueError("--questions must contain an evaluation question list")
    matches = [q for i, q in enumerate(questions)
               if str(q.get("query_id") or question_id(q, i)) == args.query_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one question {args.query_id}; found {len(matches)}")
    q = matches[0]
    if args.query_id not in annotations:
        raise ValueError(f"Query ID not found in source annotations: {args.query_id}")
    ann = annotations[args.query_id]
    if ann.get("timestamp_start") is None or ann.get("timestamp_end") is None:
        raise ValueError(f"Missing timestamp_start/end for {args.query_id}")
    start, end = float(ann["timestamp_start"]), float(ann["timestamp_end"])
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
        raise ValueError(f"Invalid timestamp interval for {args.query_id}")
    sample = ann.get("sample_name") or q.get("sample_name")
    domain = q.get("dataset") or ann.get("dataset")
    if not sample:
        raise ValueError(f"Question has no sample_name: {args.query_id}")
    mapping = load_mapping(Path(ds["video_mapping_path"]), ds.get("domains", []))
    if not domain:
        possible = [d for s, d in mapping if s == sample]
        if len(possible) != 1:
            raise ValueError(f"Dataset is ambiguous for {args.query_id}: {possible}")
        domain = possible[0]
    root = Path(ds["prepared_video_root"])
    dataset_root = Path(ds["video_mapping_path"]).resolve().parent
    available = mapping.get((sample, domain), [])
    selected = selected_views(ann, available)
    if args.view_ids:
        requested = set(args.view_ids)
        selected = [v for v in selected if v["view"] in requested]
        missing = requested - {v["view"] for v in selected}
        if missing:
            raise ValueError(f"Requested views are not designated for this question: {sorted(missing)}")
    if not selected:
        raise ValueError("No views selected")
    options = choice_options(q, ann)
    gold = q.get("correct_answer") or ann.get("correct_answer")
    if gold not in set("ABCD"):
        raise ValueError(f"Invalid/missing correct answer for {args.query_id}: {gold}")

    view_data = []
    for view in selected:
        video = mapped_video_path(root, dataset_root, view["target"])
        if not video.is_file():
            raise FileNotFoundError(video)
        view_data.append({"view_id": view["view"], "video": str(video)})
    request_spec = {"query_id": args.query_id, "question": q["question"], "options": options,
                    "correct_answer": gold, "timestamp": [start, end], "views": view_data,
                    "frames_per_view": args.frames_per_view, "point_window_seconds": args.point_window_seconds,
                    "prompt_version": "local-evidence-graph-v1", "model_path": str(config["model"]["model_path"])}
    signature = hashlib.sha256(json.dumps(request_spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    out = args.output
    manifest_path, partial_path = out / "manifest.json", out / "predictions.partial.json"
    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {manifest_path}")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("request_signature") != signature:
            raise ValueError("Resume settings differ from the saved manifest")
        records = json.loads(partial_path.read_text(encoding="utf-8")) if partial_path.is_file() else []
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(manifest_path, {"request_signature": signature, "request": request_spec,
                                   "created_at": datetime.now(timezone.utc).isoformat()})
        records = []
    runtime = get_runtime(Path(config["model"]["model_path"]))
    write_json(out / "run.json", {"status": "running", "runtime": runtime.metadata,
                                   "started_at": datetime.now(timezone.utc).isoformat(), "request_signature": signature})
    media = config.get("media", {})
    qframes: dict[str, list[dict[str, Any]]] = {}
    for item in view_data:
        vid = item["view_id"]
        task_dir = out / "frames" / f"view_{vid}"
        qframes[vid] = sample_timestamp_frames(Path(item["video"]), task_dir, start, end,
            args.frames_per_view, int(media.get("jpeg_quality", 80)), int(media.get("max_image_side", 448)),
            args.point_window_seconds)
    frame_rows = [{"view_id": vid, **frame} for vid, frames in qframes.items() for frame in frames]
    write_json(out / "sampled_frames.json", [{k: v for k, v in row.items() if k != "absolute_file"} | {"file": row["absolute_file"]}
                                             for row in frame_rows])

    def call(step_id: str, label: str, records_for_model: list[dict[str, Any]], prompt: str,
             token_limit: int) -> dict[str, Any]:
        existing = next((r for r in records if r["step_id"] == step_id and r.get("status") == "completed"), None)
        if existing:
            return existing
        task = out / "calls" / step_id
        task.mkdir(parents=True, exist_ok=True)
        (task / "prompt.txt").write_text(prompt, encoding="utf-8")
        started = time.perf_counter()
        row: dict[str, Any] = {"step_id": step_id, "label": label, "status": "running"}
        try:
            raw, metrics = runtime(records_for_model, token_limit, task, prompt)
            (task / "raw_response.txt").write_text(raw, encoding="utf-8")
            row.update(status="completed", raw_response=raw, metrics=metrics)
            try:
                row["parsed"] = parse_json_object(raw)
            except Exception as exc:
                row.update(status="parse_error", parse_error=f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        if isinstance(row.get("parsed"), dict) and "answer" in row["parsed"]:
            row["answer_letter"] = answer_letter(row["parsed"].get("answer"))
            row["correct"] = row["answer_letter"] == gold if row["answer_letter"] else None
        records.append(row)
        write_json(partial_path, records)
        print(f"{len(records)} {label}: {row['status']}", flush=True)
        return row

    try:
        # First capture visual-only baselines. These are independent answers, not the graph fusion input.
        for item in view_data:
            vid = item["view_id"]
            images = [{"absolute_file": f["absolute_file"], "label": f"View {vid}, {f['time_s']:.3f}s"}
                      for f in qframes[vid]]
            call(f"single_view_{vid}", f"single_view_{vid}", images,
                 answer_prompt(q["question"], options, [f"本组图像均来自 View {vid}。"]), args.qa_tokens)
        all_images = [{"absolute_file": f["absolute_file"], "label": f"View {vid}, {f['time_s']:.3f}s"}
                      for vid, frames in qframes.items() for f in frames]
        call("direct_all_views", "direct_all_views", all_images,
             answer_prompt(q["question"], options, ["以下图像来自同一场景的多个指定视角；标签标明来源视角。"]), args.qa_tokens)

        graph_rows = []
        for item in view_data:
            vid = item["view_id"]
            images = [{"absolute_file": f["absolute_file"], "label": f"View {vid}, frame_id={i}, {f['time_s']:.3f}s"}
                      for i, f in enumerate(qframes[vid])]
            row = call(f"local_graph_view_{vid}", f"local_graph_view_{vid}", images,
                       graph_extraction_prompt(q["question"], vid, qframes[vid]), args.graph_tokens)
            if row.get("status") != "completed":
                continue
            parsed = row.get("parsed", {})
            parsed.setdefault("view_id", vid)
            graph_rows.append(parsed)
        if len(graph_rows) == len(view_data):
            for step_id, label, dedup in (("graph_count_no_dedup", "graph_count_no_dedup", False),
                                          ("graph_count_dedup", "graph_count_dedup", True)):
                prompt = graph_fusion_prompt(q["question"], options, graph_rows, dedup)
                call(step_id, label, [], prompt, args.fusion_tokens)
        write_json(out / "predictions.json", records)
        write_json(out / "evidence_graphs.json", {"query_id": args.query_id, "question": q["question"],
            "options": options, "correct_answer": gold, "views": graph_rows})
        summary = [{"step_id": r["step_id"], "status": r.get("status"),
                    "answer": r.get("answer_letter"), "correct": r.get("correct"),
                    "elapsed_seconds": r.get("elapsed_seconds")} for r in records]
        write_json(out / "run.json", {"status": "completed", "runtime": runtime.metadata,
            "finished_at": datetime.now(timezone.utc).isoformat(), "request_signature": signature,
            "completed_steps": [r["step_id"] for r in records if r.get("status") == "completed"],
            "failed_steps": [r["step_id"] for r in records if r.get("status") != "completed"],
            "summary": summary,
            "note": "Single-case diagnostic; model-generated entities and confidence are not ground truth/calibrated."})
        print(f"Saved diagnostic to {out}", flush=True)
        return 0
    except BaseException as exc:
        write_json(out / "run.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            "request_signature": signature})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--query-id", default=DEFAULT_QUERY_ID)
    parser.add_argument("--view-ids", nargs="+", help="Optional subset of the question's designated views")
    parser.add_argument("--frames-per-view", type=int, default=2)
    parser.add_argument("--point-window-seconds", type=float, default=2.0)
    parser.add_argument("--qa-tokens", type=int, default=512)
    parser.add_argument("--graph-tokens", type=int, default=1400)
    parser.add_argument("--fusion-tokens", type=int, default=1600)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.frames_per_view, args.qa_tokens, args.graph_tokens, args.fusion_tokens) < 1:
        parser.error("frame and token limits must be positive")
    if not math.isfinite(args.point_window_seconds) or args.point_window_seconds <= 0:
        parser.error("--point-window-seconds must be positive")
    try:
        return run(args)
    except Exception as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

