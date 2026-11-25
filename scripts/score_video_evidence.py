"""Sample MVVBench candidate units and score their query relevance with a VLM."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import sys
import time
import traceback

from video_evidence import (answer_options, get_runtime, load_mapping, load_questions,
                        parse_relevance, question_id, relevance_prompt, safe_video_path,
                        sample_video_chunks, selected_views, split_for_sample)
from video_smoke import write_json


def run(config_path: Path, output: Path, extract_only: bool = False, resume: bool = False) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ds = config["dataset"]
    if ds.get("name") != "mvvbench":
        raise ValueError("dataset.name currently supports only 'mvvbench'; switch source domains in domains")
    domains = ds.get("domains", [])
    questions = load_questions(Path(ds["annotation_path"]))
    mapping = load_mapping(Path(ds["video_mapping_path"]), domains)
    split = ds.get("split", "test")
    split_cfg = ds.get("split_by_scene", {})
    seed = int(split_cfg.get("seed", 0))
    train_fraction, val_fraction = float(split_cfg.get("train", .6)), float(split_cfg.get("validation", .2))
    if resume:
        if not output.is_dir() or not (output / "run.json").is_file():
            raise FileNotFoundError("--resume requires an existing output directory with run.json")
        previous = json.loads((output / "run.json").read_text(encoding="utf-8"))
        if previous.get("config") != config:
            raise ValueError("Cannot resume: config differs from the saved run.json config")
        if previous.get("extract_only") != extract_only:
            raise ValueError("Cannot resume: extract_only mode differs from the saved run")
    else:
        output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    meta = (previous if resume else {"experiment_id": "EXP-002", "started_at": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path), "config": config, "python": sys.version, "extract_only": extract_only})
    if resume:
        meta.pop("finished_at", None)
        meta.pop("error", None)
        meta.pop("traceback", None)
    meta.update(status="running", last_resumed_at=datetime.now(timezone.utc).isoformat() if resume else None,
                resume_count=int(meta.get("resume_count", 0)) + (1 if resume else 0))
    write_json(output / "run.json", meta)
    try:
        records = []
        eval_questions = []
        seen_scenes = set()
        seen_qids = set()
        for index, row in enumerate(questions):
            sample = row.get("sample_name")
            if not isinstance(sample, str) or not sample:
                raise ValueError(f"Question row {index} missing sample_name")
            if split_for_sample(sample, seed, train_fraction, val_fraction) != split:
                continue
            qid = question_id(row, index)
            if qid in seen_qids:
                raise ValueError(f"Duplicate question id: {qid}")
            seen_qids.add(qid)
            domain = row.get("dataset")
            if not domain:
                options = [key for key in domains if (sample, key) in mapping]
                if len(options) != 1:
                    raise ValueError(f"{qid}: dataset source is ambiguous; include dataset in annotations or select one domain")
                domain = options[0]
            if domains and domain not in domains:
                continue
            available = mapping.get((sample, domain), [])
            if not available:
                continue
            scene_key = (domain, sample)
            if ds.get("one_question_per_scene", False) and scene_key in seen_scenes:
                continue
            views = selected_views(row, available)
            query = row.get("question")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"{qid}: missing question text")
            options = answer_options(row)
            correct = row.get("correct_answer")
            if not isinstance(correct, str) or correct not in options:
                raise ValueError(f"{qid}: invalid correct_answer")
            eval_questions.append({"query_id": qid, "sample_name": sample, "dataset": domain,
                                  "question": query, "options": options, "correct_answer": correct})
            records.append((qid, query, views))
            seen_scenes.add(scene_key)
        max_per_domain = ds.get("max_questions_per_domain")
        if max_per_domain is not None:
            if type(max_per_domain) is not int or max_per_domain < 1:
                raise ValueError("max_questions_per_domain must be a positive integer or null")
            counts, kept = {}, []
            for record, question in zip(records, eval_questions):
                domain = question["dataset"]
                if counts.get(domain, 0) < max_per_domain:
                    kept.append((record, question))
                    counts[domain] = counts.get(domain, 0) + 1
            records = [pair[0] for pair in kept]
            eval_questions = [pair[1] for pair in kept]
        if not records:
            raise ValueError(f"No questions found for split={split!r}, domains={domains}")

        media_cfg = config["media"]
        root = Path(ds["prepared_video_root"]).resolve()
        runtime = None if extract_only else get_runtime(Path(config["model"]["model_path"]))
        meta["runtime"] = None if extract_only else runtime.metadata
        meta["n_questions"] = len(records)
        candidates_out, assets_out = [], []
        scores_root = output / "scoring"
        for qi, (qid, query, views) in enumerate(records):
            print(f"{qi + 1}/{len(records)} scoring {qid}", flush=True)
            items, assets = [], []
            for view in views:
                video = safe_video_path(root, view["target"])
                if not video.is_file():
                    raise FileNotFoundError(video)
                view_key = f"view_{view['view']}"
                qdir = hashlib.sha256(qid.encode("utf-8")).hexdigest()[:16]
                chunks = sample_video_chunks(video, output / "assets" / qdir / view_key,
                    float(media_cfg["chunk_seconds"]), int(media_cfg["frames_per_chunk"]),
                    int(media_cfg["jpeg_quality"]), media_cfg.get("max_chunks_per_view"),
                    int(media_cfg.get("max_image_side", 448)))
                for chunk in chunks:
                    item_id = f"{view_key}_chunk_{chunk['chunk_index']:05d}"
                    files = [{**f, "file": (output / "assets" / qdir / view_key / f["file"]).resolve().as_posix()}
                             for f in chunk["files"]]
                    score, reason, metrics = None, "extract_only", None
                    if runtime is not None:
                        task_dir = scores_root / qdir / item_id
                        task_dir.mkdir(parents=True, exist_ok=True)
                        score_file = task_dir / "score.json"
                        cached = None
                        if resume and score_file.is_file():
                            try:
                                cached = json.loads(score_file.read_text(encoding="utf-8"))
                                if not isinstance(cached, dict) or "score" not in cached:
                                    cached = None
                            except (OSError, json.JSONDecodeError):
                                cached = None
                        if cached is not None:
                            score, reason, metrics = cached.get("score"), cached.get("reason"), cached.get("metrics")
                        else:
                            prompt = relevance_prompt(query)
                            (task_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
                            vlm_records = [{"absolute_file": f["file"], "label": f"View {view['view']}, time={f['time_s']:.3f}s"} for f in files]
                            raw, metrics = runtime(vlm_records, int(config["model"]["max_new_tokens"]), task_dir, prompt)
                            (task_dir / "raw_response.txt").write_text(raw, encoding="utf-8")
                            score, reason = parse_relevance(raw)
                            write_json(score_file, {"score": score, "reason": reason, "metrics": metrics})
                    item = {"item_id": item_id, "view_id": str(view["view"]), "start_s": chunk["start_s"],
                            "end_s": chunk["end_s"], "bytes": chunk["bytes"],
                            "relevance_score": score if score is not None else 0.0}
                    items.append(item)
                    assets.append({"query_id": qid, "item_id": item_id, "view_id": str(view["view"]),
                                   "start_s": chunk["start_s"], "files": files,
                                   "scoring_status": "completed" if score is not None else "not_scored",
                                   "relevance_reason": reason, "scoring_metrics": metrics})
            candidates_out.append({"query_id": qid, "items": items})
            assets_out.extend(assets)
        if not extract_only:
            write_json(output / "candidate_units.json", candidates_out)
        else:
            write_json(output / "candidate_units_unscored.json", candidates_out)
        write_json(output / "candidate_assets.json", assets_out)
        write_json(output / "evaluation_questions.json", eval_questions)
        meta.update(status="extracted" if extract_only else "scored", finished_at=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter() - started,
                    note="JPEG sampled-frame payload bytes are an offline transmission proxy, not measured network traffic.")
        return 0
    except BaseException as exc:
        meta.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        if meta.get("status") == "running":
            meta.update(status="failed", error="run exited without completion")
        meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        meta["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "run.json", meta)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extract-only", action="store_true", help="sample candidate frames but do not load a VLM")
    parser.add_argument("--resume", action="store_true", help="resume an interrupted scoring run from saved per-unit score.json files")
    args = parser.parse_args()
    try:
        return run(args.config, args.output, args.extract_only, args.resume)
    except Exception as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

