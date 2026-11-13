"""Screen MVVBench diagnostic predictions and rank cases for human review.

The generated labels are candidate signals, not ground-truth claims about
view complementarity or distraction. Review the linked predictions and frames
before interpreting any signal.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def norm_answer(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).casefold().strip().split())
    return text or None


def frame_refs(row: dict[str, Any], run_dir: Path, query_id: str) -> list[dict[str, Any]]:
    task_hash = hashlib.sha256((query_id + row.get("condition_id", "")).encode()).hexdigest()[:18]
    task_dir = run_dir / "tasks" / task_hash
    refs = []
    order_by_view: Counter[str] = Counter()
    for frame in row.get("sampled_frames", []) or []:
        view = str(frame.get("view_id", ""))
        order = order_by_view[view]
        order_by_view[view] += 1
        image_path = None
        view_dir = task_dir / f"view_{view}"
        if view_dir.is_dir():
            images = sorted(view_dir.glob("*.jpg"))
            if isinstance(order, int) and 0 <= order < len(images):
                image_path = str(images[order])
            elif len(images) == 1:
                image_path = str(images[0])
        refs.append({"view": view, "time_s": frame.get("time_s"),
                     "frame_index": frame.get("frame_index"),
                     "model_input_image": image_path})
    return refs


def condition_detail(row: dict[str, Any], run_dir: Path, query_id: str) -> dict[str, Any]:
    return {"condition_id": row.get("condition_id"), "mode": row.get("mode"),
            "status": row.get("status"), "answer": row.get("answer"),
            "correct": row.get("correct"), "confidence": row.get("confidence"),
            "selected_views": row.get("selected_views", []),
            "input_frames": row.get("input_frames"), "input_bytes": row.get("input_bytes"),
            "sampled_frames": frame_refs(row, run_dir, query_id)}


def analyze(rows: list[dict[str, Any]], run_dir: Path,
            manifest_path: Path | None = None, prompt_mode: str = "auto") -> dict[str, Any]:
    manifest = read_json(manifest_path) if manifest_path and manifest_path.is_file() else {}
    requests = manifest.get("requests", []) if isinstance(manifest, dict) else []
    questions = {str(x.get("query_id")): x for x in requests if x.get("query_id")}
    selected_prompt_mode = (manifest.get("prompt_mode", "baseline")
                            if prompt_mode == "auto" else prompt_mode)
    excluded_prompt_rows = 0
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    datasets: dict[str, str] = {}
    prompt_modes: Counter[str] = Counter()
    invalid_rows = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            invalid_rows.append({"row_index": index, "reason": "row_is_not_an_object"})
            continue
        qid = str(row.get("query_id", ""))
        if not qid or not row.get("condition_id"):
            invalid_rows.append({"row_index": index, "reason": "missing_query_or_condition_id"})
            continue
        row_prompt_mode = str(row.get("prompt_mode", "baseline"))
        if selected_prompt_mode != "all" and row_prompt_mode != selected_prompt_mode:
            excluded_prompt_rows += 1
            continue
        grouped[qid][str(row["condition_id"])].append(row)
        datasets[qid] = str(row.get("dataset", "unknown"))
        prompt_modes[str(row.get("prompt_mode", "unspecified"))] += 1

    query_details = []
    coverage: dict[str, Counter[str]] = defaultdict(Counter)
    signal_counts: Counter[str] = Counter()
    signal_by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    condition_stats: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    confidence_stats: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for qid, conditions in sorted(grouped.items()):
        qmeta = questions.get(qid, {})
        dataset = str(qmeta.get("dataset", datasets.get(qid, "unknown")))
        sample_name = qmeta.get("sample_name") or next(
            (r.get("sample_name") for rs in conditions.values() for r in rs if r.get("sample_name")), None)
        prompt = qmeta.get("question") or next(
            (r.get("question") for rs in conditions.values() for r in rs if r.get("question")), None)
        signals: list[dict[str, str]] = []
        score = 0
        selected: dict[str, dict[str, Any]] = {}
        for cid, condition_rows in conditions.items():
            coverage[cid]["queries"] += 1
            if len(condition_rows) > 1:
                signals.append({"code": "duplicate_condition", "detail": f"{cid}: {len(condition_rows)} records"})
                score += 1
            selected[cid] = condition_rows[-1]

        expected_conditions = {str(x.get("condition_id")) for x in qmeta.get("conditions", [])
                              if x.get("condition_id")}
        missing_conditions = sorted(expected_conditions - set(selected))
        if missing_conditions:
            signals.append({"code": "missing_expected_conditions",
                            "detail": ", ".join(missing_conditions)})
            score += 2

        completed = {cid: row for cid, row in selected.items()
                     if row.get("status") == "completed" and isinstance(row.get("correct"), bool)}
        for cid, row in completed.items():
            condition_stats[(dataset, cid)].append(row)
            confidence = str(row.get("confidence") or "missing").casefold()
            confidence_stats[(dataset, cid, confidence)].append(row)
        for row in selected.values():
            if row.get("status") != "completed" or not isinstance(row.get("correct"), bool):
                signals.append({"code": "incomplete_or_unscorable", "detail":
                                f"{row.get('condition_id')}: status={row.get('status')}, correct={row.get('correct')}"})
                score += 2

        full = completed.get("all_views")
        singles = {cid: row for cid, row in completed.items() if cid.startswith("single_view_")}
        leaveouts = {cid: row for cid, row in completed.items() if cid.startswith("leave_out_view_")}
        frame_rows = {cid: row for cid, row in completed.items() if cid.startswith("all_views_frames_")}

        if full and len(singles) >= 2:
            single_correct = [row for row in singles.values() if row["correct"] is True]
            single_frame_counts = ", ".join(
                f"{cid}={row.get('input_frames')}" for cid, row in sorted(singles.items()))
            if full["correct"] is True and not single_correct:
                signals.append({"code": "possible_multi_view_complementarity",
                                "detail": (f"all_views correct ({full.get('input_frames')} frames); "
                                           f"{len(singles)} available single-view conditions incorrect "
                                           f"(frames: {single_frame_counts})")})
                score += 5
            elif full["correct"] is False and single_correct:
                views = sorted({str(v) for row in single_correct for v in row.get("selected_views", [])})
                signals.append({"code": "possible_added_view_interference",
                                "detail": (f"all_views incorrect ({full.get('input_frames')} frames); "
                                           f"correct single-view condition(s) use view(s) "
                                           f"{', '.join(views) or 'unknown'} (frames: {single_frame_counts})")})
                score += 5

        if full:
            for cid, row in sorted(leaveouts.items()):
                if full["correct"] is True and row["correct"] is False:
                    signals.append({"code": "view_may_be_necessary",
                                    "detail": (f"all_views correct ({full.get('input_frames')} frames); {cid} "
                                               f"incorrect ({row.get('input_frames')} frames) after removing a view")})
                    score += 3
                elif full["correct"] is False and row["correct"] is True:
                    signals.append({"code": "removing_view_may_help",
                                    "detail": (f"all_views incorrect ({full.get('input_frames')} frames); {cid} "
                                               f"correct ({row.get('input_frames')} frames) after removing a view")})
                    score += 4

            full_answer = norm_answer(full.get("answer"))
            differing = [cid for cid, row in {**singles, **leaveouts}.items()
                         if full_answer and norm_answer(row.get("answer"))
                         and norm_answer(row.get("answer")) != full_answer]
            if differing:
                signals.append({"code": "answer_changes_across_view_sets",
                                "detail": "all_views answer differs from " + ", ".join(sorted(differing))})
                score += 2

            changed_frames = [cid for cid, row in frame_rows.items() if row["correct"] != full["correct"]]
            if changed_frames:
                signals.append({"code": "frame_count_changes_correctness", "detail": ", ".join(sorted(changed_frames))})
                score += 2

        high_wrong = [cid for cid, row in completed.items()
                      if row.get("correct") is False and str(row.get("confidence", "")).casefold() == "high"]
        if high_wrong:
            signals.append({"code": "high_confidence_wrong", "detail": ", ".join(sorted(high_wrong))})
            score += 1
        if not full:
            signals.append({"code": "missing_all_views_reference", "detail": "no scorable all_views condition"})
            score += 2
        if not signals:
            continue

        for signal in signals:
            signal_counts[signal["code"]] += 1
            signal_by_dataset[dataset][signal["code"]] += 1
        priority = "high" if score >= 5 else "medium" if score >= 2 else "low"
        relevant = {"all_views", *singles, *leaveouts, *frame_rows}
        details = [condition_detail(selected[cid], run_dir, qid)
                   for cid in sorted(relevant) if cid in selected]
        query_details.append({"query_id": qid, "dataset": dataset, "sample_name": sample_name,
                              "question": prompt, "priority": priority, "priority_score": score,
                              "signals": signals, "conditions": details})

    query_details.sort(key=lambda x: (-x["priority_score"], x["dataset"], x["query_id"]))
    condition_summary = []
    for (dataset, cid), items in sorted(condition_stats.items()):
        condition_summary.append({"dataset": dataset, "condition_id": cid, "n": len(items),
                                  "accuracy": sum(bool(x["correct"]) for x in items) / len(items),
                                  "mean_input_frames": sum(float(x.get("input_frames") or 0) for x in items) / len(items),
                                  "mean_input_bytes": sum(float(x.get("input_bytes") or 0) for x in items) / len(items)})
    confidence_summary = []
    for (dataset, cid, confidence), items in sorted(confidence_stats.items()):
        confidence_summary.append({"dataset": dataset, "condition_id": cid, "confidence": confidence,
                                   "n": len(items),
                                   "accuracy": sum(bool(x["correct"]) for x in items) / len(items)})
    return {"input_rows": len(rows), "analyzed_prompt_mode": selected_prompt_mode,
            "excluded_other_prompt_mode_rows": excluded_prompt_rows, "query_count": len(grouped),
            "queries_with_review_signals": len(query_details), "prompt_modes": dict(prompt_modes),
            "condition_coverage": [{"condition_id": cid, **dict(sorted(counts.items()))}
                                   for cid, counts in sorted(coverage.items())],
            "condition_results": condition_summary,
            "confidence_accuracy": confidence_summary,
            "signal_counts": dict(sorted(signal_counts.items())),
            "signal_counts_by_dataset": {dataset: dict(sorted(counts.items()))
                                         for dataset, counts in sorted(signal_by_dataset.items())},
            "invalid_rows": invalid_rows,
            "review_queue": query_details,
            "interpretation_note": (
                "Signals only prioritize human review. They do not establish view complementarity, "
                "causal interference, evidence visibility, or statistical significance. Predictions are "
                "comparable only when model, prompt, sampled frames, and decoding settings match.")}


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "screening_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    queue = report["review_queue"]
    fields = ["priority", "priority_score", "dataset", "sample_name", "query_id", "question", "signals"]
    with (output_dir / "review_queue.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in queue:
            writer.writerow({**item, "signals": "; ".join(
                f"{s['code']}: {s['detail']}" for s in item["signals"])})
    lines = ["# MVVBench 自动筛查结果", "", f"- 输入预测记录：{report['input_rows']}",
             f"- 题目数：{report['query_count']}",
             f"- 含复核信号的题目：{report['queries_with_review_signals']}", "", "## 信号计数", ""]
    if report["signal_counts"]:
        lines.extend(f"- `{code}`：{count} 题" for code, count in report["signal_counts"].items())
    else:
        lines.append("- 未发现可筛查信号；请确认输入预测包含可评分的视角消融条件。")
    lines += ["", "## 解释限制", "", report["interpretation_note"], "",
              "各条件的准确率、输入帧/字节统计、置信度分桶，以及题目级答案、帧时间和模型输入图像路径见 `screening_report.json`。",
              "建议先人工复核高优先级题，再覆盖不同信号类别和数据集。", ""]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="diagnostic predictions.json or predictions.partial.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        help="diagnostic_manifest.json; defaults to the predictions parent directory")
    parser.add_argument("--prompt-mode", choices=["auto", "baseline", "evidence", "all"], default="auto",
                        help="auto uses the run manifest; audit-only outputs are not QA-accuracy comparisons")
    args = parser.parse_args()
    rows = read_json(args.predictions)
    if not isinstance(rows, list):
        parser.error("predictions must be a JSON list")
    manifest = args.manifest or args.predictions.parent / "diagnostic_manifest.json"
    report = analyze(rows, args.predictions.parent, manifest, args.prompt_mode)
    write_outputs(report, args.output_dir)
    print(f"Predictions: {report['input_rows']}; queries: {report['query_count']}; "
          f"review candidates: {report['queries_with_review_signals']}")
    for signal, count in report["signal_counts"].items():
        print(f"{signal}: {count}")
    print(f"Review queue: {args.output_dir / 'review_queue.csv'}")
    print(f"Details: {args.output_dir / 'screening_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

