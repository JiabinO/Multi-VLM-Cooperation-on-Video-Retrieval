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
    return {"n_predictions": len(rows), "conditions": summary, "paired_vs_all_views": comparisons,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

