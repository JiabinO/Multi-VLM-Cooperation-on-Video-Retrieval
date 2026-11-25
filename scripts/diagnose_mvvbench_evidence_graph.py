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
        "根据给定图像回答选择题，只依据可见内容。核对题目要求并选择最合适的选项。"
        "只输出一个大写字母 A、B、C 或 D，不要输出解释、标点或其他文字。\n"
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


def make_candidate_pairs(graphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entities = []
    for graph in graphs:
        view_id = str(graph.get("view_id", "unknown"))
        for entity in graph.get("entities", []):
            local_id = str(entity.get("local_id", ""))
            if local_id:
                entities.append({"view_id": view_id, "local_id": local_id,
                                 "category": entity.get("category", "unknown"),
                                 "appearance": entity.get("appearance", ""),
                                 "position": entity.get("position", ""),
                                 "visibility": entity.get("visibility", "unknown"),
                                 "evidence": entity.get("evidence", "")})
    pairs = []
    for i, first in enumerate(entities):
        for second in entities[i + 1:]:
            if first["view_id"] == second["view_id"]:
                continue
            left, right = sorted((first, second), key=lambda x: (x["view_id"], x["local_id"]))
            pairs.append({"pair_id": f"P{len(pairs) + 1:03d}",
                          "left": {k: left[k] for k in ("view_id", "local_id")},
                          "right": {k: right[k] for k in ("view_id", "local_id")},
                          "descriptions": [{k: x[k] for k in ("category", "appearance", "position", "visibility", "evidence")}
                                           for x in (left, right)]})
    return pairs


def normalize_pair_decisions(candidate_pairs: list[dict[str, Any]],
                             response: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    raw_decisions = response.get("pair_decisions", [])
    if not isinstance(raw_decisions, list):
        raw_decisions = []
    by_id = {str(x.get("pair_id")): x for x in raw_decisions if isinstance(x, dict)}
    normalized, missing = [], []
    for pair in candidate_pairs:
        decision = by_id.get(pair["pair_id"], {})
        status = decision.get("status")
        basis = decision.get("basis", "")
        if status not in {"same", "different", "uncertain"}:
            status = "uncertain"
            basis = "模型未为该候选对给出有效判断；按 uncertain 保留。"
            missing.append(pair["pair_id"])
        normalized.append({"pair_id": pair["pair_id"],
                           "entities": [pair["left"], pair["right"]],
                           "status": status, "basis": basis})
    return normalized, missing


def entity_matching_prompt(question: str, graphs: list[dict[str, Any]],
                           candidate_pairs: list[dict[str, Any]],
                           visual_grounding: bool = False) -> str:
    grounding_instruction = (
        "同时查看附带的原始帧，并根据画面中显示器外观、相邻物体和空间布局验证候选对应；"
        "每条 same 匹配都要写出可在帧中核对的视觉依据。"
        if visual_grounding else
        "当前条件只能使用下方的文字证据图，不得假设看到了原始帧。"
    )
    return (
        "你是跨视角实体匹配器。不同视角的实体编号只在各自 view 内有效；"
        "monitor_0 这类编号在不同 view 中重复出现只是命名巧合，绝不能作为匹配依据。"
        "不要计数、判断答案或推断开关状态。只利用端侧提供的可见外观、位置、相邻实体和关系，"
        "对候选清单中的每一对都必须判断，不能省略或只返回same项。"
        "same表示支持为同一台物理设备，different表示有依据认为不是同一设备，"
        "uncertain表示无法确定。外观相似或都属于显示器不足以认定same。"
        "每个basis最多一句，并引用具体可核对线索。只输出 JSON："
        "{\"pair_decisions\":[{\"pair_id\":\"P001\",\"status\":\"same|different|uncertain\","
        "\"basis\":\"一句可核对依据\"}]}。\n"
        + grounding_instruction + "\n"
        + f"问题：{question}\n候选配对（必须逐对判断）：\n{json.dumps(candidate_pairs, ensure_ascii=False, indent=2)}"
        + f"\n各视角局部证据图：\n{json.dumps(graphs, ensure_ascii=False, indent=2)}"
    )


def count_graphs(graphs: list[dict[str, Any]], options: dict[str, str],
                 matches: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Apply transparent count rules; local IDs are always scoped by view_id."""
    nodes: dict[str, dict[str, Any]] = {}
    for graph in graphs:
        view_id = str(graph.get("view_id", "unknown"))
        for entity in graph.get("entities", []):
            local_id = str(entity.get("local_id", ""))
            if not local_id:
                continue
            key = f"{view_id}::{local_id}"
            nodes[key] = {"view_id": view_id, "local_id": local_id,
                          "state": str(entity.get("screen_state") or "uncertain"),
                          "evidence": entity.get("evidence", ""),
                          "position": entity.get("position", "")}

    parent = {key: key for key in nodes}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    applied_same_links = []
    different_links = []
    uncertain_links = []
    rejected_links = []
    for link in matches or []:
        refs = link.get("entities", [])
        keys = [f"{str(ref.get('view_id'))}::{str(ref.get('local_id'))}"
                for ref in refs if isinstance(ref, dict)]
        if len(keys) != 2 or any(key not in nodes for key in keys):
            rejected_links.append({"link": link, "reason": "unknown_or_invalid_entity_reference"})
            continue
        if nodes[keys[0]]["view_id"] == nodes[keys[1]]["view_id"]:
            rejected_links.append({"link": link, "reason": "cross_view_link_required"})
            continue
        status = link.get("status")
        if status == "same":
            a, b = find(keys[0]), find(keys[1])
            if a != b:
                a_views = {node["view_id"] for key, node in nodes.items() if find(key) == a}
                b_views = {node["view_id"] for key, node in nodes.items() if find(key) == b}
                if a_views & b_views:
                    rejected_links.append({"link": link, "reason": "would_merge_two_entities_in_same_view"})
                    continue
                parent[b] = a
            applied_same_links.append(link)
        elif status == "different":
            different_links.append(link)
        elif status == "uncertain":
            uncertain_links.append(link)
        else:
            rejected_links.append({"link": link, "reason": "invalid_match_status"})

    groups: dict[str, list[dict[str, Any]]] = {}
    for key, node in nodes.items():
        groups.setdefault(find(key), []).append(node)

    entities = []
    for group in groups.values():
        states = {node["state"] for node in group}
        known = states & {"appears_on", "appears_off"}
        if known == {"appears_on", "appears_off"}:
            fused_state = "conflict"
        elif known == {"appears_on"}:
            fused_state = "appears_on"
        elif known == {"appears_off"}:
            fused_state = "appears_off"
        else:
            fused_state = "uncertain"
        entities.append({"members": [{"view_id": n["view_id"], "local_id": n["local_id"]} for n in group],
                         "observed_states": sorted(states), "fused_state": fused_state,
                         "evidence": [{"view_id": n["view_id"], "state": n["state"],
                                       "evidence": n["evidence"], "position": n["position"]} for n in group]})

    on_count = sum(x["fused_state"] == "appears_on" for x in entities)
    off_count = sum(x["fused_state"] == "appears_off" for x in entities)
    unresolved = sum(x["fused_state"] in {"uncertain", "conflict"} for x in entities)
    possible_differences = sorted({abs((on_count + k) - (off_count + unresolved - k))
                                   for k in range(unresolved + 1)})
    option_values: dict[str, int] = {}
    for letter, text in options.items():
        match = re.fullmatch(r"\s*(\d+)\s*", str(text))
        if match:
            option_values[letter] = int(match.group(1))
    matching_answers = sorted(letter for letter, value in option_values.items()
                              if value in possible_differences)
    answer = matching_answers[0] if unresolved == 0 and len(possible_differences) == 1 and len(matching_answers) == 1 else "unknown"
    return {"entities_after_fusion": entities, "count_on": on_count, "count_off": off_count,
            "count_uncertain_or_conflict": unresolved,
            "known_state_difference": abs(on_count - off_count),
            "possible_differences_under_unresolved_states": possible_differences,
            "answer_candidates": matching_answers, "answer": answer,
            "same_links_applied": applied_same_links, "different_links": different_links,
            "uncertain_links": uncertain_links, "rejected_links": rejected_links}


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
                    "token_limits": {"qa": args.qa_tokens, "graph": args.graph_tokens, "matching": args.fusion_tokens},
                    "prompt_version": "local-evidence-graph-v4", "model_path": str(config["model"]["model_path"])}
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
             token_limit: int, answer_only: bool = False) -> dict[str, Any]:
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
                if answer_only:
                    try:
                        row["parsed"] = parse_json_object(raw)
                    except (json.JSONDecodeError, ValueError):
                        plain = raw.strip().upper()
                        if re.fullmatch(r"[ABCD]", plain):
                            row["parsed"] = {"answer": plain}
                        else:
                            raise
                else:
                    row["parsed"] = parse_json_object(raw)
            except Exception as exc:
                row.update(status="truncated" if metrics.get("reached_token_limit") else "parse_error",
                           parse_error=f"{type(exc).__name__}: {exc}")
            if metrics.get("reached_token_limit"):
                row["status"] = "truncated"
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        if isinstance(row.get("parsed"), dict) and "answer" in row["parsed"]:
            row["answer_letter"] = answer_letter(row["parsed"].get("answer"))
            row["correct"] = (row["answer_letter"] == gold
                               if row["answer_letter"] and row["status"] == "completed" else None)
        records.append(row)
        write_json(partial_path, records)
        print(f"{len(records)} {label}: {row['status']}", flush=True)
        return row

    def save_derived(step_id: str, result: dict[str, Any]) -> None:
        answer = result.get("answer")
        row = {"step_id": step_id, "label": step_id,
               "status": result.get("status", "completed"), "result": result,
               "answer_letter": answer if answer in set("ABCD") else None,
               "correct": answer == gold if answer in set("ABCD") else None}
        existing_i = next((i for i, old in enumerate(records) if old.get("step_id") == step_id), None)
        if existing_i is None:
            records.append(row)
        else:
            records[existing_i] = row
        write_json(partial_path, records)
        print(f"{step_id}: {row['status']} counts=" + json.dumps(
            {k: result.get(k) for k in ("count_on", "count_off", "count_uncertain_or_conflict",
                                         "known_state_difference", "answer_candidates", "answer")},
            ensure_ascii=False), flush=True)

    try:
        # First capture visual-only baselines. These are independent answers, not the graph fusion input.
        for item in view_data:
            vid = item["view_id"]
            images = [{"absolute_file": f["absolute_file"], "label": f"View {vid}, {f['time_s']:.3f}s"}
                      for f in qframes[vid]]
            call(f"single_view_{vid}", f"single_view_{vid}", images,
                 answer_prompt(q["question"], options, [f"本组图像均来自 View {vid}。"]), args.qa_tokens,
                 answer_only=True)
        all_images = [{"absolute_file": f["absolute_file"], "label": f"View {vid}, {f['time_s']:.3f}s"}
                      for vid, frames in qframes.items() for f in frames]
        call("direct_all_views", "direct_all_views", all_images,
             answer_prompt(q["question"], options, ["以下图像来自同一场景的多个指定视角；标签标明来源视角。"]),
             args.qa_tokens, answer_only=True)

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
            no_dedup = count_graphs(graph_rows, options)
            save_derived("graph_count_no_dedup", no_dedup)
            candidate_pairs = make_candidate_pairs(graph_rows)
            write_json(out / "candidate_entity_pairs.json", {"candidate_pairs": candidate_pairs})
            for suffix, use_visual in (("text", False), ("visual", True)):
                match_step = f"cross_view_entity_matching_{suffix}"
                match_inputs = []
                if use_visual:
                    match_inputs = [
                        {"absolute_file": qframes[item["view_id"]][0]["absolute_file"],
                         "label": f"Original grounding frame for View {item['view_id']}"}
                        for item in view_data
                    ]
                match_row = call(match_step, match_step, match_inputs,
                    entity_matching_prompt(q["question"], graph_rows, candidate_pairs,
                                           visual_grounding=use_visual),
                    args.fusion_tokens)
                match_data = match_row.get("parsed", {}) if match_row.get("status") == "completed" else {}
                match_list, missing_pairs = normalize_pair_decisions(candidate_pairs, match_data)
                decision_rows = match_data.get("pair_decisions", [])
                if not isinstance(decision_rows, list):
                    decision_rows = []
                write_json(out / f"cross_view_matches_{suffix}.json", {
                    "candidate_pairs": candidate_pairs, "matches": match_list,
                    "missing_pair_decisions": missing_pairs,
                    "invalid_decisions": [x for x in decision_rows
                                          if isinstance(x, dict) and x.get("pair_id") not in
                                          {p["pair_id"] for p in candidate_pairs}],
                    "raw_response": match_row.get("raw_response"),
                    "status": match_row.get("status"), "visual_grounding": use_visual})
                derived_id = f"graph_count_dedup_{suffix}"
                if match_row.get("status") == "completed":
                    save_derived(derived_id, count_graphs(graph_rows, options, match_list))
                else:
                    save_derived(derived_id, {"status": "skipped_match_step_" + str(match_row.get("status"))})
        elif graph_rows:
            save_derived("graph_count_no_dedup", {"status": "skipped_missing_view_graphs",
                         "available_view_graphs": len(graph_rows), "expected_view_graphs": len(view_data)})
            save_derived("graph_count_dedup", {"status": "skipped_missing_view_graphs"})
        write_json(out / "predictions.json", records)
        write_json(out / "evidence_graphs.json", {"query_id": args.query_id, "question": q["question"],
            "options": options, "correct_answer": gold, "views": graph_rows})
        summary = [{"step_id": r["step_id"], "status": r.get("status"),
                    "answer": r.get("answer_letter"), "correct": r.get("correct"),
                    "elapsed_seconds": r.get("elapsed_seconds"),
                    "counts": r.get("result")} for r in records]
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
    parser.add_argument("--qa-tokens", type=int, default=64)
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

