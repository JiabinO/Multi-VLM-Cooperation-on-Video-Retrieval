"""Summarize paired EXP-002 predictions and measured upload bytes."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize(plans: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    plan_map = {}
    for plan in plans:
        key = (plan["query_id"], plan["policy"], float(plan["budget_fraction"]))
        if key in plan_map:
            raise ValueError("duplicate plan condition")
        plan_map[key] = plan
    pred_map = {}
    for pred in predictions:
        key = (pred["query_id"], pred["policy"], float(pred["budget_fraction"]))
        if key in pred_map:
            raise ValueError("duplicate prediction condition")
        if key not in plan_map:
            raise ValueError("prediction has no matching plan")
        answer = pred.get("answer")
        feasible = plan_map[key].get("feasible", True)
        if feasible and not isinstance(answer, str):
            raise ValueError("feasible plan predictions require an answer string")
        if not feasible and answer is not None:
            raise ValueError("infeasible plan prediction answer must be null")
        pred_map[key] = pred
    if not pred_map or pred_map.keys() != plan_map.keys():
        raise ValueError("predictions and plans must have identical condition keys")

    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    domain_groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for key, plan in plan_map.items():
        pred = pred_map[key]
        # Recompute per condition; the validation loop above also uses a
        # local named `feasible`, which must not leak into this aggregation loop.
        feasible = plan.get("feasible", True) and pred.get("status") != "infeasible_budget"
        gold = pred.get("correct_answer")
        if not isinstance(gold, str) or not gold.strip():
            raise ValueError("each prediction row requires correct_answer for analysis")
        expected_bytes = plan["uplink_bytes"]
        measured_bytes = pred.get("uplink_bytes", expected_bytes)
        if type(measured_bytes) is not int or measured_bytes < 0:
            raise ValueError("uplink_bytes must be a nonnegative integer")
        norm = lambda value: " ".join(value.casefold().split())
        row = {"query_id": plan["query_id"], "policy": plan["policy"],
               "budget_fraction": float(plan["budget_fraction"]),
               "dataset": pred.get("dataset", "unknown"),
               "correct": norm(pred["answer"]) == norm(gold) if feasible else None,
               "feasible": feasible,
               "planned_bytes": expected_bytes, "uplink_bytes": measured_bytes,
               "budget_bytes": plan["budget_bytes"], "full_input_bytes": plan["full_input_bytes"]}
        if measured_bytes > plan["budget_bytes"] and plan["policy"] != "full_input":
            raise ValueError("measured upload exceeds configured per-query budget")
        groups[(plan["policy"], float(plan["budget_fraction"]))].append(row)
        domain_groups[(row["dataset"], plan["policy"], float(plan["budget_fraction"]))].append(row)

    def condition_rows(grouped, domain_key=False):
        conditions = []
        for key, rows in sorted(grouped.items()):
            if domain_key:
                domain, policy, fraction = key
            else:
                policy, fraction = key
                domain = None
            n = len(rows)
            evaluated = [row for row in rows if row["feasible"]]
            item = {
                "policy": policy, "budget_fraction": fraction, "n_queries": n,
                "n_evaluated": len(evaluated),
                "accuracy": (sum(row["correct"] for row in evaluated) / len(evaluated) if evaluated else None),
                "mean_uplink_bytes": sum(row["uplink_bytes"] for row in rows) / n,
                "mean_full_input_bytes": sum(row["full_input_bytes"] for row in rows) / n,
                "mean_budget_utilization": (sum(row["uplink_bytes"] for row in rows) /
                                             sum(row["budget_bytes"] for row in rows)
                                             if policy != "full_input" and sum(row["budget_bytes"] for row in rows) else None),
            }
            if domain is not None:
                item["dataset"] = domain
            conditions.append(item)
        return conditions

    conditions = condition_rows(groups)
    per_dataset = condition_rows(domain_groups, domain_key=True)
    return {"n_queries": len({row["query_id"] for row in plan_map.values()}),
            "conditions": conditions, "per_dataset": per_dataset}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
        report = summarize(load(args.plans), load(args.predictions))
        with Path(args.output).open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        print(f"Wrote report to {args.output}")
        return 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
