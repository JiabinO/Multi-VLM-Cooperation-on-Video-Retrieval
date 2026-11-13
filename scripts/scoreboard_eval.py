"""Validate per-frame scoreboard readings against partially reviewed references."""
import json
import re


def parse_readings(raw, frame_ids):
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("frames"), list):
        raise ValueError("Expected frames list")
    seen = set()
    for row in data["frames"]:
        if not isinstance(row, dict):
            raise ValueError("Each frame must be an object")
        fid = row.get("frame_id")
        if type(fid) is not int or fid not in frame_ids or fid in seen:
            raise ValueError("Unknown, duplicate, or noninteger frame_id")
        seen.add(fid)
        if row.get("status") not in ("readable", "unreadable", "not_visible"):
            raise ValueError("Invalid status")
        if not isinstance(row.get("uncertainty"), str):
            raise ValueError("Missing uncertainty string")
        for key in ("eng_score", "den_score"):
            value = row.get(key)
            if key not in row or (value is not None and (type(value) is not int or value < 0)):
                raise ValueError(f"Invalid {key}")
        clock = row.get("match_clock")
        if "match_clock" not in row or (clock is not None and
                (not isinstance(clock, str) or not re.fullmatch(r"\d{1,3}:[0-5]\d", clock))):
            raise ValueError("Invalid match_clock (expected M:SS or null)")
        if row["status"] == "readable" and any(row[k] is None for k in ("eng_score", "den_score", "match_clock")):
            raise ValueError("Readable requires both scores and clock")
        if row["status"] == "not_visible" and any(row[k] is not None for k in ("eng_score", "den_score", "match_clock")):
            raise ValueError("not_visible requires null values")
    if seen != set(frame_ids):
        raise ValueError("Must return exactly one row per input frame")
    return data


def evaluate(readings, reference):
    rows = {r["frame_id"]: r for r in readings["frames"]}
    details = []
    for ref in reference["frames"]:
        if ref["review_status"] != "confirmed" or "expected" not in ref:
            continue
        pred = rows.get(ref["frame_id"])
        if pred is None:
            raise ValueError("Confirmed reference frame absent from predictions")
        checks = {}
        for key, expected in ref["expected"].items():
            actual = pred.get(key)
            if key == "match_clock" and actual is not None:
                actual = tuple(map(int, actual.split(":")))
                expected = tuple(map(int, expected.split(":")))
            checks[key] = actual == expected
        details.append({"frame_id": ref["frame_id"], "expected": ref["expected"],
                        "predicted": pred, "checks": checks, "exact_match": all(checks.values())})
    n = len(details)
    return {"scope": "Only confirmed scoreboard references; other frames require human review",
            "scored_frames": n, "exact_match_frames": sum(r["exact_match"] for r in details),
            "exact_match_rate": sum(r["exact_match"] for r in details) / n if n else None,
            "unscored_frame_ids": sorted(set(rows) - {r["frame_id"] for r in details}), "details": details}
