"""Summarize timestamp-oracle MVVBench diagnostic predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in rows if r.get("status") == "completed"]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    by_q_cond = {}
    for row in rows:
        groups.setdefault((row["mode"], row["condition_id"], row.get("dataset", "unknown")), []).append(row)
        by_q_cond[(row["query_id"], row["condition_id"])] = row
    summary = []
    for (mode, condition, domain), items in sorted(groups.items()):
        summary.append({"mode": mode, "condition_id": condition, "dataset": domain,
            "n": len(items), "accuracy": sum(bool(x["correct"]) for x in items) / len(items),
            "mean_input_frames": sum(x["input_frames"] for x in items) / len(items),
            "mean_input_bytes": sum(x["input_bytes"] for x in items) / len(items)})

    comparisons = []
    conditions = sorted({r["condition_id"] for r in rows})
    reference = "all_views"
    for condition in conditions:
        if condition == reference:
            continue
        paired = [(by_q_cond[(qid, reference)], by_q_cond[(qid, condition)])
                  for qid, cond in by_q_cond if cond == condition and (qid, reference) in by_q_cond]
        if not paired:
            continue
        wins = sum(a["correct"] and not b["correct"] for a, b in paired)
        losses = sum(not a["correct"] and b["correct"] for a, b in paired)
        comparisons.append({"reference": reference, "condition_id": condition, "n_paired": len(paired),
            "reference_accuracy": sum(a["correct"] for a, _ in paired) / len(paired),
            "condition_accuracy": sum(b["correct"] for _, b in paired) / len(paired),
            "accuracy_difference": sum(int(b["correct"]) - int(a["correct"]) for a, b in paired) / len(paired),
            "reference_wins": wins, "ties": len(paired) - wins - losses, "condition_wins": losses})

    # Compare the multi-view input against the strongest single-view result for
    # each same query. This is an oracle upper bound, not an executable policy.
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_query.setdefault(row["query_id"], []).append(row)
    best_single_details = []
    for qid, items in by_query.items():
        all_views = next((x for x in items if x["condition_id"] == "all_views"), None)
        singles = [x for x in items if x["mode"] == "view_ablation"
                   and len(x.get("selected_views", [])) == 1]
        if all_views is None or not singles:
            continue
        best = max(singles, key=lambda x: (bool(x["correct"]), -int(x["input_bytes"])))
        best_single_details.append({"query_id": qid, "dataset": all_views.get("dataset", "unknown"),
            "all_views_correct": bool(all_views["correct"]),
            "best_single_correct": bool(best["correct"]),
            "best_single_view": best["selected_views"][0],
            "best_single_bytes": best["input_bytes"],
            "all_views_answer": all_views.get("answer"), "best_single_answer": best.get("answer")})

    def paired_summary(details: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not details:
            return None
        all_acc = sum(x["all_views_correct"] for x in details) / len(details)
        single_acc = sum(x["best_single_correct"] for x in details) / len(details)
        all_only = sum(x["all_views_correct"] and not x["best_single_correct"] for x in details)
        single_only = sum(x["best_single_correct"] and not x["all_views_correct"] for x in details)
        return {"n_paired": len(details), "all_views_accuracy": all_acc,
                "best_single_oracle_accuracy": single_acc,
                "accuracy_difference_all_minus_best_single": all_acc - single_acc,
                "all_views_only_correct": all_only,
                "best_single_only_correct": single_only,
                "same_correctness": len(details) - all_only - single_only}

    best_single = {"overall": paired_summary(best_single_details),
        "by_dataset": {domain: paired_summary([x for x in best_single_details if x["dataset"] == domain])
                       for domain in sorted({x["dataset"] for x in best_single_details})},
        "per_query": best_single_details,
        "interpretation": "Best single view is chosen using the answer key; use only as a diagnostic upper bound."}

    frame_sweeps = []
    for nframes in sorted({int(r["frames_per_view"]) for r in rows
                           if r["condition_id"].startswith("all_views_frames_")}):
        condition = f"all_views_frames_{nframes}"
        paired = [(by_q_cond[(qid, "all_views")], by_q_cond[(qid, condition)])
                  for qid in by_query if (qid, "all_views") in by_q_cond
                  and (qid, condition) in by_q_cond]
        if paired:
            frame_sweeps.append({"frames_per_view": nframes, "n_paired": len(paired),
                "all_views_4frame_accuracy": sum(bool(a["correct"]) for a, _ in paired) / len(paired),
                "accuracy": sum(bool(b["correct"]) for _, b in paired) / len(paired),
                "accuracy_difference": sum(int(bool(b["correct"])) - int(bool(a["correct"]))
                                           for a, b in paired) / len(paired),
                "all_views_wins": sum(bool(b["correct"]) and not bool(a["correct"]) for a, b in paired),
                "frame_condition_wins": sum(bool(a["correct"]) and not bool(b["correct"]) for a, b in paired),
                "ties": sum(bool(a["correct"]) == bool(b["correct"]) for a, b in paired)})

    return {"n_predictions": len(rows), "conditions": summary, "paired_vs_all_views": comparisons,
            "best_single_view_oracle": best_single, "paired_frame_sweep_vs_4frames": frame_sweeps,
            "note": "Oracle timestamp diagnostic; results are conditional on the annotated temporal window."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.predictions.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        parser.error("predictions must be a JSON list")
    report = analyze(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Completed predictions: {report['n_predictions']}; condition summaries: {len(report['conditions'])}")
    for item in report["conditions"]:
        print(f"{item['dataset']} | {item['condition_id']} | n={item['n']} | "
              f"acc={item['accuracy']:.3f} | frames={item['mean_input_frames']:.1f} | "
              f"bytes={item['mean_input_bytes']:.0f}")
    best = report["best_single_view_oracle"]["overall"]
    if best:
        print("paired all_views vs per-question best single-view oracle: "
              f"n={best['n_paired']} all={best['all_views_accuracy']:.3f} "
              f"best_single={best['best_single_oracle_accuracy']:.3f} "
              f"all_only={best['all_views_only_correct']} "
              f"single_only={best['best_single_only_correct']}")
    for item in report["paired_frame_sweep_vs_4frames"]:
        print(f"paired frames/view={item['frames_per_view']} vs 4: "
              f"n={item['n_paired']} acc={item['accuracy']:.3f} "
              f"delta={item['accuracy_difference']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

