"""Shared MVVBench loading, media sampling, and local Qwen helpers for EXP-002."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any


def load_questions(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        if isinstance(value.get("data"), list):
            value = value["data"]
        elif isinstance(value.get("questions"), list):
            value = value["questions"]
        else:
            # MVVBench's original JSON groups question rows by sample name.
            rows = []
            for sample_name, group in value.items():
                if not isinstance(group, list):
                    continue
                for item in group:
                    if isinstance(item, dict):
                        row = dict(item)
                        if not row.get("sample_name"):
                            row["sample_name"] = str(sample_name)
                        rows.append(row)
            value = rows
    if not isinstance(value, list) or not value or not all(isinstance(x, dict) for x in value):
        raise ValueError("Expected MVVBench question list or sample-keyed object")
    return value

def question_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("question_id") or row.get("id") or f"{row['sample_name']}-q{index:05d}")


def answer_options(row: dict[str, Any]) -> dict[str, str]:
    source = row.get("choices")
    if not isinstance(source, dict):
        source = row
    result = {}
    for letter in "ABCD":
        value = source.get(letter)
        if value is None:
            value = source.get(f"choice_{letter}")
        if isinstance(value, str) and value.strip():
            result[letter] = value
    if len(result) < 2:
        raise ValueError("Question must have at least two nonempty answer choices")
    return result

def load_mapping(path: Path, domains: list[str]) -> dict[tuple[str, str], list[dict[str, str]]]:
    result: dict[tuple[str, str], list[dict[str, str]]] = {}
    seen: set[tuple[str, str, str]] = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            domain = row.get("dataset", "")
            if domains and domain not in domains:
                continue
            sample, view, target = row.get("sample_name"), row.get("view"), row.get("target")
            if not sample or view is None or not target:
                raise ValueError("Invalid video_mapping.csv row")
            identity = (sample, domain, str(view))
            if identity in seen:
                raise ValueError(f"Duplicate sample/domain/view mapping: {identity}")
            seen.add(identity)
            result.setdefault((sample, domain), []).append({"view": str(view), "target": target})
    return result


def selected_views(row: dict[str, Any], available: list[dict[str, str]]) -> list[dict[str, str]]:
    # `views` defines the benchmark's designated candidate input set; it is not
    # passed to the relevance-scoring or QA prompt.
    requested = row.get("views")
    if not isinstance(requested, list) or not requested:
        raise ValueError("Question needs a nonempty designated views list")
    requested = {str(v) for v in requested}
    found = [x for x in available if x["view"] in requested]
    if {x["view"] for x in found} != requested:
        raise ValueError(f"Missing mapped views: {sorted(requested - {x['view'] for x in found})}")
    return sorted(found, key=lambda x: x["view"])


def split_for_sample(sample: str, seed: int, train: float, validation: float) -> str:
    if not 0 < train < 1 or not 0 < validation < 1 or train + validation >= 1:
        raise ValueError("train/validation fractions must be positive and sum to less than one")
    digest = hashlib.sha256(f"{seed}:{sample}".encode()).digest()
    point = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if point < train else "validation" if point < train + validation else "test"


def safe_video_path(root: Path, target: str) -> Path:
    rel = PurePosixPath(target)
    if rel.is_absolute() or ".." in rel.parts or "\\" in target or ":" in target:
        raise ValueError("Unsafe video mapping target")
    path = (root / Path(*rel.parts)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Video path escapes configured root")
    return path


def sample_video_chunks(video: Path, out_dir: Path, chunk_seconds: float, frame_count: int,
                        jpeg_quality: int, max_chunks: int | None = None,
                        max_image_side: int = 448) -> list[dict[str, Any]]:
    import cv2
    from PIL import Image

    if not math.isfinite(chunk_seconds) or chunk_seconds <= 0 or frame_count < 1:
        raise ValueError("chunk_seconds and frame_count must be positive")
    if type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be an integer from 1 to 100")
    if type(max_image_side) is not int or max_image_side < 32:
        raise ValueError("max_image_side must be an integer of at least 32")
    if max_chunks is not None and (type(max_chunks) is not int or max_chunks < 1):
        raise ValueError("max_chunks_per_view must be a positive integer or null")
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        fps, nframes = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or nframes <= 0:
            raise ValueError(f"Invalid FPS/frame count: {video}")
        duration = nframes / fps
        count = max(1, math.ceil(duration / chunk_seconds))
        if max_chunks and count > max_chunks:
            indices = sorted({round(i * (count - 1) / (max_chunks - 1)) for i in range(max_chunks)}) if max_chunks > 1 else [count // 2]
        else:
            indices = list(range(count))
        result = []
        for chunk_index in indices:
            start, end = chunk_index * chunk_seconds, min((chunk_index + 1) * chunk_seconds, duration)
            first = min(nframes - 1, max(0, math.ceil(start * fps)))
            last = min(nframes - 1, max(first, math.ceil(end * fps) - 1))
            sample_n = min(frame_count, last - first + 1)
            frame_ids = [first] if sample_n == 1 else [round(first + i * (last - first) / (sample_n - 1)) for i in range(sample_n)]
            chunk_dir = out_dir / f"chunk_{chunk_index:05d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            files, total_bytes = [], 0
            for frame_no, frame_index in enumerate(frame_ids):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f"Frame decode failed: {video} at {frame_index}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                image.thumbnail((max_image_side, max_image_side), Image.Resampling.LANCZOS)
                rel = Path(f"chunk_{chunk_index:05d}") / f"frame_{frame_no:02d}.jpg"
                target = out_dir / rel
                image.save(target, format="JPEG", quality=jpeg_quality, optimize=False)
                size = target.stat().st_size
                total_bytes += size
                files.append({"file": rel.as_posix(), "bytes": size, "time_s": frame_index / fps})
            result.append({"chunk_index": chunk_index, "start_s": start, "end_s": end,
                           "files": files, "bytes": total_bytes})
        return result
    finally:
        cap.release()


def get_runtime(model_path: Path):
    from w02_qa import QwenRuntime
    return QwenRuntime(model_path)


def relevance_prompt(question: str) -> str:
    return ("你是多摄像头视频问答系统的端侧证据打分器。给定一个问题和单个摄像头的一小段视频帧，判断该片段对回答问题的证据价值。"
            "只根据可见内容评分；看不清或无关则给低分。不要推断其他摄像头或未出现的时段，不要猜正确选项。"
            "score 是 0 到 1 的数字。仅输出 JSON：{\"score\":0.0,\"reason\":\"简短可见依据或不确定原因\"}。\n问题：" + question)


def parse_relevance(raw: str) -> tuple[float, str]:
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    value = json.loads(text)
    score = value.get("score") if isinstance(value, dict) else None
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("VLM relevance score must be numeric in [0,1]")
    reason = value.get("reason", "")
    if not isinstance(reason, str):
        raise ValueError("VLM relevance reason must be text")
    return float(score), reason


def qa_prompt(question: str, options: dict[str, str]) -> str:
    option_text = "\n".join(f"{key}. {value}" for key, value in options.items())
    return ("你将看到一个问题允许使用的多视角视频证据帧。综合所有输入帧，选择最有证据支持的一个选项。"
            "若证据不充分，仍选择最可能选项。不要使用外部知识。只输出 JSON：{\"answer\":\"A\",\"confidence\":\"low|medium|high\"}。\n"
            f"问题：{question}\n选项：\n{option_text}")


def parse_answer(raw: str, valid_options: dict[str, str] | set[str] | None = None) -> tuple[str, str]:
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    value = json.loads(text)
    answer = value.get("answer") if isinstance(value, dict) else None
    confidence = value.get("confidence", "") if isinstance(value, dict) else ""
    allowed = set(valid_options) if valid_options is not None else set("ABCD")
    if answer not in allowed or confidence not in ("low", "medium", "high"):
        raise ValueError(f"Expected answer in {sorted(allowed)} and confidence low/medium/high")
    return answer, confidence
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    value = json.loads(text)
    answer = value.get("answer") if isinstance(value, dict) else None
    confidence = value.get("confidence", "") if isinstance(value, dict) else ""
    if answer not in {"A", "B", "C", "D"} or confidence not in ("low", "medium", "high"):
        raise ValueError("Expected answer A-D and confidence low/medium/high")
    return answer, confidence
