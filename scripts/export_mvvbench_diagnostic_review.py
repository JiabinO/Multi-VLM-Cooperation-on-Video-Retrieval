"""Export full-resolution audit frames for discordant MVVBench diagnoses.

These files are for human inspection only. They do not change the images used
for model inference or any recorded predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def export(run_dir: Path, output: Path, jpeg_quality: int = 95) -> int:
    import cv2
    from PIL import Image

    report = read_json(run_dir / "report.json")
    predictions = read_json(run_dir / "predictions.json")
    manifest = read_json(run_dir / "diagnostic_manifest.json")
    details = report["best_single_view_oracle"]["per_query"]
    discordant = {x["query_id"]: (
        "single_only" if x["best_single_correct"] and not x["all_views_correct"] else "all_only")
        for x in details if x["best_single_correct"] != x["all_views_correct"]}
    questions = {x["query_id"]: x for x in manifest["requests"]}
    predictions_by_query: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        predictions_by_query.setdefault(row["query_id"], []).append(row)
    output.mkdir(parents=True, exist_ok=True)
    rows, key_rows = [], []
    for qid, review_group in sorted(discordant.items()):
        q = questions[qid]
        qpreds = predictions_by_query[qid]
        all_row = next(x for x in qpreds if x["condition_id"] == "all_views")
        best_view = next(d["best_single_view"] for d in details if d["query_id"] == qid)
        best_condition = "single_view_" + str(best_view)
        best_row = next(x for x in qpreds if x["condition_id"] == best_condition)
        task_hash = hashlib.sha256((qid + "all_views").encode()).hexdigest()[:18]
        lowres_task = run_dir / "tasks" / task_hash
        view_paths = {str(v["view"]): Path(v["path"]) for v in q["views"]}
        frame_order: dict[str, int] = {}
        highres_paths: list[str] = []
        model_paths: list[str] = []
        for frame in all_row["sampled_frames"]:
            view = str(frame["view_id"])
            order = frame_order.get(view, 0)
            frame_order[view] = order + 1
            source = view_paths[view]
            cap = cv2.VideoCapture(str(source))
            try:
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open source video: {source}")
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame["frame_index"]))
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode {source} frame {frame['frame_index']}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                target_dir = output / hashlib.sha256(qid.encode()).hexdigest()[:12] / f"view_{view}"
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / f"frame_{order:03d}_at_{frame['time_s']:.3f}s.jpg"
                image.save(target, format="JPEG", quality=jpeg_quality)
                highres_paths.append(str(target))
                lowres = lowres_task / f"view_{view}" / f"frame_{order:03d}.jpg"
                model_paths.append(str(lowres))
            finally:
                cap.release()
        rows.append({"query_id": qid, "dataset": q["dataset"],
            "sample_name": q["sample_name"], "question": q["question"],
            "options": json.dumps(q["options"], ensure_ascii=False),
            "timestamp_start": q["timestamp_start"], "timestamp_end": q["timestamp_end"],
            "high_resolution_frames": "\n".join(highres_paths),
            "model_input_frames": "\n".join(model_paths)})
        key_rows.append({"review_group": review_group, "query_id": qid,
            "all_views_answer": all_row["answer"], "all_views_correct": all_row["correct"],
            "best_single_view": best_row["selected_views"][0],
            "best_single_answer": best_row["answer"], "best_single_correct": best_row["correct"],
            "correct_answer": q["correct_answer"]})
    index = output / "review_index_blind.csv"
    fields = list(rows[0]) if rows else ["query_id", "dataset", "sample_name", "question", "options",
        "timestamp_start", "timestamp_end", "high_resolution_frames", "model_input_frames"]
    with index.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    key_path = output / "review_answer_key.csv"
    with key_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(key_rows[0]) if key_rows else [
            "review_group", "query_id", "all_views_answer", "all_views_correct", "best_single_view",
            "best_single_answer", "best_single_correct", "correct_answer"])
        writer.writeheader()
        writer.writerows(key_rows)
    (output / "README.txt").write_text(
        "review_index_blind.csv lists questions/options and audit images without predictions or gold answers.\n"
        "Record your blind review before opening review_answer_key.csv.\n"
        "Full-resolution frames are from exact source indices used by the diagnostic; compare them with model_input_frames.\n"
        "Audit images are not model inputs and do not alter predictions. For motion, inspect the source clip at the timestamp window.\n",
        encoding="utf-8")
    print(f"Exported {len(rows)} discordant queries; blind index: {index}; answer key: {key_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        help="defaults to RUN_DIR/review_fullres")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")
    output = args.output or args.run_dir / "review_fullres"
    try:
        return export(args.run_dir, output, args.jpeg_quality)
    except Exception as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

