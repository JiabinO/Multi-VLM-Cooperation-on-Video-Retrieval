"""Map Assembly101 event labels to nominal clip frames and build a small review set."""
import argparse
import csv
from pathlib import Path


FPS = 30
CLIP_FRAMES = 150

# One representative event/view per window. W04 includes the two frames that
# correspond to sparse observation sample IDs 4 and 5 (of 8) for a direct check.
REVIEW_EVENTS = (
    ("W03", "C10095", "unscrew chassis with screwdriver", (0, 60, 119, 120, 121)),
    ("W04", "C10118", "remove chassis part", (85, 106, 127, 135, 136, 142, 149)),
    ("W05", "C10115", "pick up screw", (14, 17, 24, 32, 40, 49, 55)),
    ("W06", "C10118", "pick up excavator arm", (60, 65, 66, 70, 74, 79, 80)),
)


def align_row(row, fps=FPS, clip_frames=CLIP_FRAMES):
    """Return a copy with global event and candidate overlap mapped to clip frames."""
    start_frame = int(row["event_start_frame_30fps"])
    end_frame = int(row["event_end_frame_30fps"])
    window_start_frame = round(float(row["window_start_sec_nominal"]) * fps)
    overlap_start = int(row["candidate_overlap_start_frame"]) - window_start_frame
    overlap_end = int(row["candidate_overlap_end_frame"]) - window_start_frame
    local_start = start_frame - window_start_frame
    local_end = end_frame - window_start_frame
    result = dict(row)
    result.update({
        "nominal_window_start_frame_30fps": window_start_frame,
        "event_start_frame_local_nominal": local_start,
        "event_end_frame_local_nominal": local_end,
        "candidate_overlap_start_frame_local_nominal": overlap_start,
        "candidate_overlap_end_frame_local_nominal": overlap_end,
        "clip_intersection_start_frame_0based": max(0, overlap_start),
        "clip_intersection_end_frame_0based": min(clip_frames - 1, overlap_end),
        "clip_overlap_status": (
            "no_overlap" if overlap_end < 0 or overlap_start >= clip_frames else
            "starts_before_clip" if local_start < 0 else
            "extends_after_clip" if local_end >= clip_frames else
            "within_clip"
        ),
        "alignment_status": "nominal_offset_only_pending_visual_review",
    })
    return result


def _read_candidates(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "window_id", "granularity", "view_id", "action_cls",
            "event_start_frame_30fps", "event_end_frame_30fps",
            "candidate_overlap_start_frame", "candidate_overlap_end_frame",
            "window_start_sec_nominal",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Candidate CSV missing columns: {sorted(required - set(reader.fieldnames or []))}")
        return list(reader.fieldnames), list(reader)


def _write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_review_rows(aligned_rows):
    review = []
    for window, view, action, indices in REVIEW_EVENTS:
        matches = [row for row in aligned_rows if (
            row["window_id"] == window and row["granularity"] == "fine" and
            row["view_id"] == view and row["action_cls"] == action
        )]
        if len(matches) != 1:
            raise ValueError(f"Expected one candidate for {window}/{view}/{action}, found {len(matches)}")
        event = matches[0]
        for index in indices:
            if not 0 <= index < CLIP_FRAMES:
                raise ValueError(f"Review frame out of range: {window} frame {index}")
            purpose = "sample_id_4_or_5_comparison" if window == "W04" and index in (85, 106) else "event_boundary_audit"
            review.append({
                "window_id": window,
                "view_id": view,
                "action_cls": action,
                "event_start_frame_global_30fps": event["event_start_frame_30fps"],
                "event_end_frame_global_30fps": event["event_end_frame_30fps"],
                "event_start_frame_local_nominal": event["event_start_frame_local_nominal"],
                "event_end_frame_local_nominal": event["event_end_frame_local_nominal"],
                "clip_frame_0based": index,
                "clip_time_sec_nominal": f"{index / FPS:.3f}",
                "review_purpose": purpose,
                "human_observation": "",
                "alignment_verdict": "pending",
            })
    return review


def extract_contact_sheets(review_rows, clips_dir, output_dir):
    import cv2
    import numpy as np

    clips_dir, output_dir = Path(clips_dir), Path(output_dir)
    frames_dir = output_dir / "selected_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for window, view, action, _ in REVIEW_EVENTS:
        clip = clips_dir / f"{window}_{view}.mp4"
        if not clip.is_file():
            raise FileNotFoundError(f"Missing clip: {clip}")
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open clip: {clip}")
        items = [r for r in review_rows if r["window_id"] == window]
        tiles = []
        for item in items:
            frame_index = int(item["clip_frame_0based"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(f"Cannot decode {clip} frame {frame_index}")
            cv2.putText(frame, f"clip frame {frame_index} | {frame_index / FPS:.2f}s",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)
            image_path = frames_dir / f"{window}_{view}_f{frame_index:03d}.jpg"
            if not cv2.imwrite(str(image_path), frame):
                cap.release()
                raise RuntimeError(f"Cannot write {image_path}")
            tiles.append(frame)
        cap.release()
        width = max(tile.shape[1] for tile in tiles)
        height = max(tile.shape[0] for tile in tiles)
        gap = 8
        title_height = 28
        cols = min(4, len(tiles))
        rows = (len(tiles) + cols - 1) // cols
        sheet = np.full((title_height + rows * height + (rows + 1) * gap,
                         cols * width + (cols + 1) * gap, 3), 245, dtype=np.uint8)
        for i, tile in enumerate(tiles):
            y = title_height + gap + (i // cols) * (height + gap)
            x = gap + (i % cols) * (width + gap)
            sheet[y:y + height, x:x + width] = tile
        title = f"{window} {view}: {action} | nominal event frames " + \
                f"{items[0]['event_start_frame_local_nominal']}..{items[0]['event_end_frame_local_nominal']}"
        cv2.putText(sheet, title[:120], (gap, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (25, 25, 25), 1, cv2.LINE_AA)
        target = output_dir / f"{window}_{view}_contact_sheet.jpg"
        if not cv2.imwrite(str(target), sheet):
            raise RuntimeError(f"Cannot write {target}")


def write_review_notes(path, candidate_count, review_rows):
    text = f"""# Assembly101 事件帧对齐人工抽查

本批将 {candidate_count} 条官方事件候选按窗口名义起始时间和 30 fps 映射到 5 秒片段；映射是待目视确认的名义值，不代表逐帧同步已被证明。片段按 150 帧、0 起始编号处理，事件端点保留官方给出的帧号约定。

公式：`片段帧号 = 官方事件帧号 - round(窗口起始秒 × 30)`。审核图上标的是片段内 0-based 帧号及时间。`review_subset.csv` 记录了每个审核帧。

## 建议先看这四组

- W03 / C10095：螺丝刀拧螺丝事件；帧 0、60、119–121 用来确认事件持续状态及官方 104 秒附近的动作切换。
- W04 / C10118：拆下底盘部件；官方起始帧映射到片段帧 136，事件终点映射到 190，已超出这段片段。额外列出片段帧 85、106，分别对应 `video_smoke.py` 8 帧均匀采样中的第 4、5 个采样点（0 起始 ID）。请重点核对此前人工记录的“第 5、6 帧开始拆卸”是否与官方时间段一致；它可能反映时间边界差异或帧号口径差异。
- W05 / C10115：拿起螺丝事件；检查事件映射帧 17–49 前后。
- W06 / C10118：拿起挖掘机臂事件；官方区间只有 8 帧，检查 66–74 附近是否能在单帧中辨认动作。低帧率抽样可能漏掉该短事件。

请在 `review_subset.csv` 的 `human_observation` 和 `alignment_verdict` 列填写审核结果。建议判定 `aligned`、`offset_suspected`、`event_label_unclear` 或 `cannot_tell`，并在备注中写出实际可见动作及大致首末帧。

## 重要边界

你已确认所有窗口三路视频的片段帧 0 与源视频窗口起点画面一致，这支持窗口起点和摄像头对应关系。它尚不能单独证明片段全程时间戳严格一致；重编码、抽帧和官方端点语义都可能带来少量偏差。这个抽查用于决定是否接受名义映射或需要修正，当前不应把映射当作已验证真值。
"""
    Path(path).write_text(text, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="event_window_candidates.csv")
    parser.add_argument("--clips", help="Directory containing W03-W06 per-view clips")
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--mapping-only", action="store_true", help="Write tables without opening video clips")
    args = parser.parse_args(argv)
    if not args.mapping_only and not args.clips:
        parser.error("--clips is required unless --mapping-only is set")

    fields, candidates = _read_candidates(args.candidates)
    aligned = [align_row(row) for row in candidates]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    aligned_fields = fields + [
        "nominal_window_start_frame_30fps", "event_start_frame_local_nominal",
        "event_end_frame_local_nominal", "candidate_overlap_start_frame_local_nominal",
        "candidate_overlap_end_frame_local_nominal", "clip_intersection_start_frame_0based",
        "clip_intersection_end_frame_0based", "clip_overlap_status", "alignment_status",
    ]
    _write_csv(output / "aligned_event_candidates.csv", aligned, aligned_fields)
    review_rows = make_review_rows(aligned)
    _write_csv(output / "review_subset.csv", review_rows, list(review_rows[0]))
    write_review_notes(output / "README.md", len(aligned), review_rows)
    if not args.mapping_only:
        extract_contact_sheets(review_rows, args.clips, output)
    print(f"Aligned {len(aligned)} candidate rows; selected {len(review_rows)} review frames.")
    print(f"Output: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
