"""Import official Assembly101 action labels and prepare review tables (stdlib only)."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

FPS = 30
DEFAULT_VIEWS = ("C10095", "C10115", "C10118")
DEFAULT_PROFILES = (8, 24)
DEFAULT_WINDOWS = {
    "W03": {"start_sec": 100, "end_sec": 105},
    "W04": {"start_sec": 140, "end_sec": 145},
    "W05": {"start_sec": 180, "end_sec": 185},
    "W06": {"start_sec": 220, "end_sec": 225},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame(value, field, source):
    try:
        frame = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} in {source}: {value!r}") from exc
    if frame < 0:
        raise ValueError(f"Negative {field} in {source}: {frame}")
    return frame


def parse_fine_csv(paths, session_id, views=DEFAULT_VIEWS):
    """Read official fine-grained CSVs; return only exact session RGB view rows."""
    events = []
    seen = set()
    prefix = session_id + "/"
    for path in paths:
        path = Path(path)
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"video", "start_frame", "end_frame", "action_cls", "verb_cls", "noun_cls"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"Unexpected fine-grained CSV schema in {path}; expected {sorted(required)}")
            for row in reader:
                video = (row.get("video") or "").strip()
                if not video.startswith(prefix):
                    continue
                basename = video.rsplit("/", 1)[-1]
                match = re.fullmatch(r"(C[0-9]+)_rgb\.mp4", basename)
                if not match or match.group(1) not in views:
                    continue
                start = _frame(row.get("start_frame"), "start_frame", path)
                end = _frame(row.get("end_frame"), "end_frame", path)
                if end < start:
                    raise ValueError(f"Reversed fine-grained segment in {path}: {start}>{end}")
                event = {
                    "event_id": f"fine:{video}:{start}:{end}:{row.get('action_id', '')}",
                    "granularity": "fine",
                    "source_file": path.name,
                    "sequence_id": session_id,
                    "video": video,
                    "view_id": match.group(1),
                    "start_frame_30fps": start,
                    "end_frame_30fps": end,
                    "start_sec_30fps": start / FPS,
                    "end_sec_30fps": end / FPS,
                    "action_id": row.get("action_id", ""),
                    "verb_id": row.get("verb_id", ""),
                    "noun_id": row.get("noun_id", ""),
                    "action_cls": (row.get("action_cls") or "").strip(),
                    "verb_cls": (row.get("verb_cls") or "").strip(),
                    "noun_cls": (row.get("noun_cls") or "").strip(),
                    "toy_id": row.get("toy_id", ""),
                    "toy_name": row.get("toy_name", ""),
                    "label_end_frame_semantics": "preserved_as_published; inclusive/exclusive convention not inferred",
                }
                if event["event_id"] in seen:
                    continue
                seen.add(event["event_id"])
                events.append(event)
    return events


def _coarse_fields(line):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "," in line or "\t" in line:
        try:
            fields = next(csv.reader([line], delimiter="," if "," in line else "\t"))
        except csv.Error:
            fields = []
    else:
        fields = line.split(maxsplit=2)
    if len(fields) < 3:
        return None
    if fields[0].strip().lower() in {"start_frame", "start"}:
        return None
    return [part.strip() for part in fields[:3]]


def parse_coarse_labels(paths, session_id):
    """Parse official per-sequence coarse label text files (start, end, class)."""
    events = []
    for path in paths:
        path = Path(path)
        expected = re.fullmatch(r"(assembly|disassembly)_" + re.escape(session_id) + r"\.txt", path.name)
        if not expected:
            raise ValueError(f"Coarse label filename does not match session {session_id}: {path.name}")
        with path.open("r", encoding="utf-8-sig") as stream:
            for line_no, line in enumerate(stream, 1):
                fields = _coarse_fields(line)
                if fields is None:
                    continue
                start = _frame(fields[0], "start_frame", path)
                end = _frame(fields[1], "end_frame", path)
                if end < start:
                    raise ValueError(f"Reversed coarse segment at {path}:{line_no}")
                action = fields[2].strip()
                if not action:
                    raise ValueError(f"Empty coarse action at {path}:{line_no}")
                events.append({
                    "event_id": f"coarse:{expected.group(1)}:{line_no}:{start}:{end}",
                    "granularity": "coarse",
                    "source_file": path.name,
                    "sequence_id": session_id,
                    "phase": expected.group(1),
                    "line_number": line_no,
                    "start_frame_30fps": start,
                    "end_frame_30fps": end,
                    "start_sec_30fps": start / FPS,
                    "end_sec_30fps": end / FPS,
                    "action_cls": action,
                    "label_end_frame_semantics": "preserved_as_published; inclusive/exclusive convention not inferred",
                })
    return events


def load_windows(path=None):
    windows = json.loads(Path(path).read_text(encoding="utf-8")) if path else DEFAULT_WINDOWS
    if not isinstance(windows, dict) or not windows:
        raise ValueError("Windows must be a nonempty object")
    result = {}
    for name, bounds in windows.items():
        if not isinstance(bounds, dict):
            raise ValueError(f"Invalid bounds for {name}")
        start, end = bounds.get("start_sec"), bounds.get("end_sec")
        if (not isinstance(start, (int, float)) or isinstance(start, bool) or
                not isinstance(end, (int, float)) or isinstance(end, bool) or start < 0 or end <= start):
            raise ValueError(f"Invalid time range for {name}")
        result[name] = {"start_sec": start, "end_sec": end,
                        "start_frame_30fps": int(start * FPS), "end_frame_30fps": int(end * FPS)}
    return result


def window_candidates(events, windows):
    """Conservatively join annotation frame spans to nominal source-time windows."""
    rows = []
    for event in events:
        for window_id, window in windows.items():
            start, end = event["start_frame_30fps"], event["end_frame_30fps"]
            w_start, w_end = window["start_frame_30fps"], window["end_frame_30fps"]
            overlap_start = max(start, w_start)
            overlap_end = min(end, w_end)
            if overlap_start > overlap_end:
                continue
            rows.append({
                "event_id": event["event_id"],
                "window_id": window_id,
                "granularity": event["granularity"],
                "view_id": event.get("view_id", "all_views_coarse"),
                "action_cls": event["action_cls"],
                "event_start_frame_30fps": start,
                "event_end_frame_30fps": end,
                "event_start_sec_30fps": event["start_sec_30fps"],
                "event_end_sec_30fps": event["end_sec_30fps"],
                "candidate_overlap_start_frame": overlap_start,
                "candidate_overlap_end_frame": overlap_end,
                "window_start_sec_nominal": window["start_sec"],
                "window_end_sec_nominal": window["end_sec"],
                "mapping_status": "candidate_only_source_pts_and_clip_offset_unverified",
            })
    return rows


def evidence_review_rows(candidates, views=DEFAULT_VIEWS, profiles=DEFAULT_PROFILES):
    rows = []
    for candidate in candidates:
        for view in views:
            for profile in profiles:
                rows.append({
                    "event_id": candidate["event_id"],
                    "window_id": candidate["window_id"],
                    "granularity": candidate["granularity"],
                    "action_cls": candidate["action_cls"],
                    "required_fact_unit": "",
                    "view_id": view,
                    "sampling_profile_frames": profile,
                    "visibility": "unchecked",
                    "support_frame_ids": "",
                    "reviewer_note": "",
                })
    return rows


def _write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def build_manifest(fine_paths, coarse_paths, session_id, windows_path, out, views=DEFAULT_VIEWS):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    fine_events = parse_fine_csv(fine_paths, session_id, views)
    coarse_events = parse_coarse_labels(coarse_paths, session_id)
    events = sorted(fine_events + coarse_events,
                    key=lambda e: (e["start_frame_30fps"], e["granularity"], e.get("view_id", ""), e["event_id"]))
    windows = load_windows(windows_path)
    candidates = window_candidates(events, windows)
    checks = evidence_review_rows(candidates, views)
    sources = [*map(Path, fine_paths), *map(Path, coarse_paths)]
    manifest = {
        "schema_version": 1,
        "status": "official_labels_imported_pending_visual_review",
        "dataset": "Assembly101",
        "sequence_id": session_id,
        "annotation_fps": FPS,
        "view_ids": list(views),
        "sampling_profiles_frames_per_view": list(DEFAULT_PROFILES),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "source_files": [{"name": p.name, "sha256": sha256(p)} for p in sources],
        "time_mapping_note": "frame/FPS retained as nominal annotation time; source video PTS and clip offsets are not verified",
        "end_frame_note": "published end frame retained; inclusive/exclusive semantics are not inferred",
        "events": events,
        "event_window_candidates": candidates,
    }
    (out / "annotation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(out / "event_window_candidates.csv", candidates)
    _write_csv(out / "evidence_review_template.csv", checks)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--fine-csv", type=Path, nargs="+", required=True,
                        help="Official train/validation/test fine-grained CSV file(s)")
    parser.add_argument("--coarse-label", type=Path, nargs="*", default=[],
                        help="Matching assembly_...txt and/or disassembly_...txt")
    parser.add_argument("--windows", type=Path, help="JSON object with start_sec/end_sec per window")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; must not exist")
    args = parser.parse_args()
    result = build_manifest(args.fine_csv, args.coarse_label, args.session_id, args.windows, args.output)
    print(json.dumps({"status": result["status"], "events": len(result["events"]),
                      "window_candidates": len(result["event_window_candidates"]),
                      "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
