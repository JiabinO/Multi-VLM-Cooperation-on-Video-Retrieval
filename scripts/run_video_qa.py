"""Run selected MVVBench video evidence plans through the configured VLM."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import sys
import time
import traceback

from video_evidence import get_runtime, parse_answer, qa_prompt
from video_smoke import write_json


def run(plans_path: Path, assets_path: Path, questions_path: Path, config_path: Path,
        output: Path, policies: list[str] | None = None, fractions: list[float] | None = None,
        resume: bool = False) -> int:
    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    qmap = {q["query_id"]: q for q in questions}
    amap = {(a["query_id"], a["item_id"]): a for a in assets}
    if len(qmap) != len(questions) or len(amap) != len(assets):
        raise ValueError("Duplicate question or candidate asset IDs")
    selected_plans = [p for p in plans if (policies is None or p["policy"] in policies)
                      and (fractions is None or float(p["budget_fraction"]) in fractions)]
    if not selected_plans:
        raise ValueError("No plans match requested policies/budget fractions")

    def condition_key(row):
        return (row["query_id"], row["policy"], float(row["budget_fraction"]))

    predictions_path = output / "predictions.partial.json"
    if resume:
        if not output.is_dir():
            raise FileNotFoundError(f"Cannot resume: output directory does not exist: {output}")
        saved_plans_path = output / "plans.json"
        if not saved_plans_path.is_file():
            raise FileNotFoundError(f"Cannot resume: missing saved plan manifest: {saved_plans_path}")
        saved_plans = json.loads(saved_plans_path.read_text(encoding="utf-8"))
        requested_by_key = {condition_key(p): p for p in selected_plans}
        saved_by_key = {condition_key(p): p for p in saved_plans}
        if requested_by_key != saved_by_key or len(requested_by_key) != len(selected_plans):
            raise ValueError("Resume plans do not match the plans saved in the output directory")

        # A completed run can also be resumed safely: use its final predictions
        # as the checkpoint when the partial checkpoint is absent.
        checkpoint = predictions_path if predictions_path.is_file() else output / "predictions.json"
        predictions = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.is_file() else []
        seen = set()
        for prediction in predictions:
            key = condition_key(prediction)
            if key not in saved_by_key:
                raise ValueError(f"Checkpoint contains a prediction outside the saved plans: {key}")
            if key in seen:
                raise ValueError(f"Checkpoint contains a duplicate prediction: {key}")
            seen.add(key)
        pending_plans = [p for p in selected_plans if condition_key(p) not in seen]
        print(f"Resuming: {len(predictions)} completed, {len(pending_plans)} remaining", flush=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plans.json", selected_plans)
        predictions = []
        pending_plans = selected_plans

    meta = {"status": "running", "experiment_id": "EXP-002", "started_at": datetime.now(timezone.utc).isoformat(),
            "config": config, "python": sys.version, "n_conditions": len(selected_plans),
            "resume": resume, "n_already_completed": len(predictions)}
    write_json(output / "run.json", meta)
    started = time.perf_counter()
    cache = {}
    try:
        if pending_plans:
            runtime = get_runtime(Path(config["model"]["model_path"]))
            meta["runtime"] = runtime.metadata
        else:
            runtime = None
        for index, plan in enumerate(pending_plans):
            qid = plan["query_id"]
            if qid not in qmap:
                raise ValueError(f"No question record for {qid}")
            chosen = plan["selected_item_ids"]
            if not plan.get("feasible", True):
                q = qmap[qid]
                predictions.append({"query_id": qid, "policy": plan["policy"],
                    "budget_fraction": float(plan["budget_fraction"]), "answer": None,
                    "dataset": q.get("dataset"), "correct_answer": q["correct_answer"],
                    "confidence": None, "uplink_bytes": 0, "planned_bytes": 0,
                    "selected_item_ids": [], "status": "infeasible_budget",
                    "reason": plan.get("infeasible_reason")})
                write_json(output / "predictions.partial.json", predictions)
                print(f"{len(predictions)}/{len(selected_plans)} {qid} {plan['policy']} {plan['budget_fraction']}: infeasible", flush=True)
                continue
            key = (qid, tuple(sorted(chosen)))
            metrics, raw, answer, confidence, reused = None, None, None, None, False
            if key in cache:
                raw, answer, confidence, metrics = cache[key]
                reused = True
            else:
                item_records = []
                for item_id in chosen:
                    asset = amap.get((qid, item_id))
                    if asset is None:
                        raise ValueError(f"Missing candidate asset {qid}/{item_id}")
                    if asset.get("scoring_status") != "completed":
                        raise ValueError(f"Candidate {qid}/{item_id} was not VLM-scored")
                    files = asset["files"]
                    for f in files:
                        item_records.append({"absolute_file": f["file"],
                            "label": f"View {asset['view_id']}, time={f['time_s']:.3f}s",
                            "view_id": asset["view_id"], "time_s": f["time_s"]})
                if not item_records:
                    raise ValueError(f"No selected frames for {qid}/{plan['policy']}")
                # Maintain a stable chronological order regardless of greedy selection rank.
                item_records.sort(key=lambda x: (x["view_id"], x["time_s"], x["absolute_file"]))
                qdir = hashlib.sha256(qid.encode("utf-8")).hexdigest()[:16]
                task = output / "tasks" / qdir / plan["policy"] / f"budget_{float(plan['budget_fraction']):.3f}"
                task.mkdir(parents=True, exist_ok=True)
                q = qmap[qid]
                prompt = qa_prompt(q["question"], q["options"])
                (task / "prompt.txt").write_text(prompt, encoding="utf-8")
                raw, metrics = runtime(item_records, int(config["model"]["qa_max_new_tokens"]), task, prompt)
                (task / "raw_response.txt").write_text(raw, encoding="utf-8")
                answer, confidence = parse_answer(raw)
                cache[key] = (raw, answer, confidence, metrics)
            q = qmap[qid]
            predictions.append({"query_id": qid, "policy": plan["policy"],
                "budget_fraction": float(plan["budget_fraction"]), "answer": answer,
                "dataset": q.get("dataset"),
                "correct_answer": q["correct_answer"], "confidence": confidence,
                "uplink_bytes": int(plan["uplink_bytes"]), "planned_bytes": int(plan["uplink_bytes"]),
                "selected_item_ids": chosen, "status": "completed", "raw_response": raw, "metrics": metrics,
                "reused_identical_input": reused})
            write_json(output / "predictions.partial.json", predictions)
            print(f"{len(predictions)}/{len(selected_plans)} {qid} {plan['policy']} {plan['budget_fraction']}: {answer}", flush=True)
        if len(predictions) != len(selected_plans):
            raise RuntimeError(f"Run produced {len(predictions)} of {len(selected_plans)} planned predictions")
        write_json(output / "predictions.json", predictions)
        meta.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter() - started,
                    note="uplink_bytes is the sum of selected JPEG payload bytes; no network transfer was measured.")
        return 0
    except BaseException as exc:
        meta.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        meta.setdefault("status", "failed")
        meta.setdefault("finished_at", datetime.now(timezone.utc).isoformat())
        meta["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "run.json", meta)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", choices=["full_input", "uniform_temporal_sampling", "fixed_view_sampling", "query_driven_selection"])
    parser.add_argument("--fractions", nargs="+", type=float)
    parser.add_argument("--resume", action="store_true",
                        help="resume this output directory from predictions.partial.json; without this flag, start fresh")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        return run(args.plans, args.assets, args.questions, args.config, args.output,
                   args.policies, args.fractions, args.resume)
    except Exception as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
