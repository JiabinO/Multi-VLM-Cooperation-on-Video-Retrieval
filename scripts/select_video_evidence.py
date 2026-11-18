"""Create deterministic per-query transmission plans for EXP-002.

This script operates on pre-measured candidate units. It does not decode videos
or infer relevance scores; a separate runtime selector must produce those
scores from the query and permitted video content, without MVVBench answer-side
annotations.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _number(value: Any, name: str, *, nonnegative: bool = True) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def validate_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_queries: set[str] = set()
    for row in rows:
        qid = row.get("query_id")
        if not isinstance(qid, str) or not qid.strip() or qid in seen_queries:
            raise ValueError("query_id must be nonempty and unique")
        seen_queries.add(qid)
        items = row.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError(f"{qid}: items must be a nonempty list")
        seen_items: set[str] = set()
        for item in items:
            iid, view = item.get("item_id"), item.get("view_id")
            if not isinstance(iid, str) or not iid or iid in seen_items:
                raise ValueError(f"{qid}: item_id must be nonempty and unique")
            if not isinstance(view, str) or not view:
                raise ValueError(f"{qid}/{iid}: view_id is required")
            seen_items.add(iid)
            cost = item.get("bytes")
            if type(cost) is not int or cost <= 0:
                raise ValueError(f"{qid}/{iid}: bytes must be a positive integer")
            start, end = _number(item.get("start_s"), "start_s"), _number(item.get("end_s"), "end_s")
            if end <= start:
                raise ValueError(f"{qid}/{iid}: end_s must be greater than start_s")
            _number(item.get("relevance_score"), "relevance_score")
        # Validate the ordering metadata used by the uniform baseline.
        if len({(x["view_id"], x["start_s"], x["end_s"]) for x in items}) != len(items):
            raise ValueError(f"{qid}: duplicate view/time candidate")
    if not rows:
        raise ValueError("candidate file is empty")
    return rows


def _uniform_order(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin temporal order across views, then chronological within view."""
    by_view: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_view.setdefault(item["view_id"], []).append(item)
    for values in by_view.values():
        values.sort(key=lambda x: (x["start_s"], x["end_s"], x["item_id"]))
    views = sorted(by_view)
    ordered: list[dict[str, Any]] = []
    level = 0
    while True:
        added = False
        for view in views:
            if level < len(by_view[view]):
                ordered.append(by_view[view][level])
                added = True
        if not added:
            return ordered
        level += 1


POLICIES = ("full_input", "uniform_temporal_sampling", "fixed_view_sampling", "query_driven_selection")


def select(items: list[dict[str, Any]], policy: str, budget_bytes: int) -> list[str]:
    if policy == "full_input":
        chosen = list(items)
    elif policy == "uniform_temporal_sampling":
        chosen = []
        used = 0
        for item in _uniform_order(items):
            if used + item["bytes"] <= budget_bytes:
                chosen.append(item)
                used += item["bytes"]
    elif policy == "fixed_view_sampling":
        # Deterministic camera-only baseline: use the lowest ordered view ID.
        view = min(item["view_id"] for item in items)
        chosen, used = [], 0
        for item in sorted((x for x in items if x["view_id"] == view),
                           key=lambda x: (x["start_s"], x["end_s"], x["item_id"])):
            if used + item["bytes"] <= budget_bytes:
                chosen.append(item)
                used += item["bytes"]
    elif policy == "query_driven_selection":
        # Deterministic budgeted maximum-density greedy approximation.
        ranked = sorted(items, key=lambda x: (
            -(x["relevance_score"] / x["bytes"]), -x["relevance_score"],
            x["start_s"], x["view_id"], x["item_id"],
        ))
        chosen = []
        used = 0
        for item in ranked:
            if used + item["bytes"] <= budget_bytes:
                chosen.append(item)
                used += item["bytes"]
    else:
        raise ValueError(f"Unknown policy: {policy}")
    return [item["item_id"] for item in chosen]


def build_plans(rows: list[dict[str, Any]], fractions: list[float], policies: list[str] | None = None) -> list[dict[str, Any]]:
    validate_candidates(rows)
    if not fractions or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 < x <= 1 for x in fractions):
        raise ValueError("fractions must be a nonempty list in (0, 1]")
    policies = policies or list(POLICIES)
    if not policies or len(policies) != len(set(policies)) or not set(policies) <= set(POLICIES):
        raise ValueError(f"policies must be unique members of {POLICIES}")
    output = []
    for row in rows:
        full_bytes = sum(item["bytes"] for item in row["items"])
        for fraction in fractions:
            budget = int(math.floor(full_bytes * fraction))
            for policy in policies:
                ids = select(row["items"], policy, budget)
                by_id = {item["item_id"]: item for item in row["items"]}
                sent = sum(by_id[i]["bytes"] for i in ids)
                if sent > budget and policy != "full_input":
                    raise AssertionError("selector exceeded budget")
                output.append({
                    "query_id": row["query_id"], "policy": policy,
                    "budget_fraction": float(fraction), "full_input_bytes": full_bytes,
                    "budget_bytes": budget, "selected_item_ids": ids,
                    "uplink_bytes": sent, "feasible": policy == "full_input" or bool(ids),
                    "infeasible_reason": None if policy == "full_input" or ids else "budget_below_smallest_candidate_unit",
                })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="JSON list; see docs/EXP-002 candidate schema")
    parser.add_argument("--config", help="Optional EXP-002 JSON config for budget fractions")
    parser.add_argument("--fractions", nargs="+", type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        rows = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
        fractions = args.fractions
        policies = None
        if fractions is None and args.config:
            config = json.loads(Path(args.config).read_text(encoding="utf-8"))
            fractions = config["budget"]["fractions_of_full_input"]
        if args.config:
            config = json.loads(Path(args.config).read_text(encoding="utf-8"))
            policies = config.get("evaluation", {}).get("methods")
        if fractions is None:
            raise ValueError("provide --fractions or --config")
        plans = build_plans(rows, fractions, policies)
        target = Path(args.output)
        with target.open("x", encoding="utf-8") as stream:
            json.dump(plans, stream, ensure_ascii=False, indent=2)
        print(f"Wrote {len(plans)} plans to {target}")
        return 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
