#!/usr/bin/env python3
"""Create a deterministic dataset/type-stratified MVVBench prompt pilot sample."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Any

from diagnose_mvvbench_evidence import annotation_index


CHALLENGE_IDS = {"64pm_industry_safety_4-q01281", "63am_industry_safety_2-q01180"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True,
                        help="evaluation_questions.json from the scoring run")
    parser.add_argument("--annotations", type=Path, required=True,
                        help="MVVBench annotation JSON configured for this dataset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--include-challenge-cases", action="store_true")
    args = parser.parse_args()
    if args.n < 2:
        parser.error("--n must be at least 2 to cover both datasets")

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    annotations = annotation_index(args.annotations)
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    missing_type: list[str] = []
    for q in questions:
        qid = q.get("query_id")
        ann = annotations.get(qid)
        if ann is None:
            raise ValueError(f"Question missing from annotation file: {qid}")
        if not args.include_challenge_cases and qid in CHALLENGE_IDS:
            continue
        dataset = str(q.get("dataset") or ann.get("dataset") or "unknown")
        kind = str(ann.get("question_type") or ann.get("category") or "unknown").strip().casefold()
        if kind == "unknown":
            missing_type.append(qid)
        view_count = len(ann.get("views", []))
        enriched = dict(q)
        enriched["_sampling_stratum"] = {"question_type": kind, "view_count": view_count}
        groups[dataset][kind].append(enriched)
    if missing_type:
        print(f"Warning: {len(missing_type)} questions lack question_type/category; grouped as unknown.")

    rng = random.Random(args.seed)
    for type_groups in groups.values():
        for items in type_groups.values():
            rng.shuffle(items)
    domains = sorted(groups)
    if len(domains) < 2:
        parser.error(f"Expected questions from at least two datasets; found {domains}")
    selected: list[dict[str, Any]] = []
    selected_per_domain: Counter[str] = Counter()
    selected_per_stratum: Counter[tuple[str, str]] = Counter()
    while len(selected) < args.n:
        made_progress = False
        for domain in domains:
            if len(selected) >= args.n:
                break
            available = [kind for kind, items in groups[domain].items() if items]
            if not available:
                continue
            kind = min(available, key=lambda k: (selected_per_stratum[(domain, k)], k))
            item = groups[domain][kind].pop()
            selected.append(item)
            selected_per_domain[domain] += 1
            selected_per_stratum[(domain, kind)] += 1
            made_progress = True
        if not made_progress:
            break
    if len(selected) < args.n:
        print(f"Warning: only {len(selected)} eligible questions available (requested {args.n}).")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.output_dir / "selected_questions.json"
    selected_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "seed": args.seed, "requested_n": args.n, "selected_n": len(selected),
        "challenge_cases_excluded": not args.include_challenge_cases,
        "selected_query_ids": [q["query_id"] for q in selected],
        "counts_by_dataset": dict(selected_per_domain),
        "counts_by_dataset_and_type": {f"{d}|{k}": n for (d, k), n in sorted(selected_per_stratum.items())},
        "counts_by_type": dict(Counter(q["_sampling_stratum"]["question_type"] for q in selected)),
        "view_count_distribution": dict(Counter(str(q["_sampling_stratum"]["view_count"]) for q in selected)),
    }
    (args.output_dir / "sampling_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Selected questions: {selected_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

