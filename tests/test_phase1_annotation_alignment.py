import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import phase1_annotation_alignment as alignment


class AlignmentTests(unittest.TestCase):
    def test_maps_official_frames_to_local_clip_frames(self):
        row = {
            "event_start_frame_30fps": "4336",
            "event_end_frame_30fps": "4390",
            "candidate_overlap_start_frame": "4336",
            "candidate_overlap_end_frame": "4350",
            "window_start_sec_nominal": "140",
        }
        result = alignment.align_row(row)
        self.assertEqual(result["event_start_frame_local_nominal"], 136)
        self.assertEqual(result["event_end_frame_local_nominal"], 190)
        self.assertEqual(result["clip_intersection_start_frame_0based"], 136)
        self.assertEqual(result["clip_intersection_end_frame_0based"], 149)
        self.assertEqual(result["clip_overlap_status"], "extends_after_clip")
        self.assertEqual(result["alignment_status"], "nominal_offset_only_pending_visual_review")

    def test_legacy_sparse_sample_ids_map_to_expected_clip_frames(self):
        sparse = [round(i * (alignment.CLIP_FRAMES - 1) / 7) for i in range(8)]
        self.assertEqual(sparse, [0, 21, 43, 64, 85, 106, 128, 149])
        self.assertEqual(sparse[4:6], [85, 106])

    def test_review_subset_has_four_windows_and_only_valid_frames(self):
        rows = []
        for window, view, action, _ in alignment.REVIEW_EVENTS:
            start = {"W03": -60, "W04": 136, "W05": 17, "W06": 66}[window]
            window_base = {"W03": 3000, "W04": 4200, "W05": 5400, "W06": 6600}[window]
            duration = {"W03": 180, "W04": 54, "W05": 32, "W06": 8}[window]
            rows.append({
                "window_id": window, "granularity": "fine", "view_id": view,
                "action_cls": action, "event_start_frame_30fps": str(start + window_base),
                "event_end_frame_30fps": str(start + duration + window_base),
                "candidate_overlap_start_frame": str(max(0, start) + window_base),
                "candidate_overlap_end_frame": str(min(149, start + duration) + window_base),
                "window_start_sec_nominal": str({"W03": 100, "W04": 140, "W05": 180, "W06": 220}[window]),
            })
        aligned = [alignment.align_row(row) for row in rows]
        review = alignment.make_review_rows(aligned)
        self.assertEqual({row["window_id"] for row in review}, {"W03", "W04", "W05", "W06"})
        self.assertTrue(all(0 <= int(row["clip_frame_0based"]) < 150 for row in review))
        w04_legacy = [row for row in review if row["window_id"] == "W04" and row["review_purpose"] == "sample_id_4_or_5_comparison"]
        self.assertEqual([int(row["clip_frame_0based"]) for row in w04_legacy], [85, 106])


if __name__ == "__main__":
    unittest.main()
