import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from phase1_annotations import build_manifest, parse_coarse_labels, parse_fine_csv


SESSION = "nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724"


class AnnotationImportTests(unittest.TestCase):
    def make_fine(self, path):
        fields = ["video", "start_frame", "end_frame", "action_id", "verb_id", "noun_id",
                  "action_cls", "verb_cls", "noun_cls", "toy_id", "toy_name"]
        rows = [
            [f"{SESSION}/C10095_rgb.mp4", "3000", "3030", "1", "2", "3", "unscrew screw",
             "unscrew", "screw", "a01", "bulldozer"],
            [f"{SESSION}/C10115_rgb.mp4", "3031", "3060", "4", "5", "6", "remove screw",
             "remove", "screw", "a01", "bulldozer"],
            [f"{SESSION}/C10095_rgb.mp4", "0", "10", "7", "8", "9", "wrong sequence",
             "verb", "noun", "a01", "bulldozer"],
            [f"other_sequence/C10095_rgb.mp4", "3000", "3030", "1", "2", "3", "wrong session",
             "verb", "noun", "a01", "bulldozer"],
            [f"{SESSION}/C99999_rgb.mp4", "3000", "3030", "1", "2", "3", "wrong view",
             "verb", "noun", "a01", "bulldozer"],
        ]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)

    def test_fine_csv_filters_exact_session_and_requested_rgb_views(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "train.csv"
            self.make_fine(source)
            events = parse_fine_csv([source], SESSION)
        self.assertEqual([e["view_id"] for e in events], ["C10095", "C10115", "C10095"])
        self.assertEqual(events[0]["start_sec_30fps"], 100)
        self.assertEqual(events[0]["action_cls"], "unscrew screw")
        self.assertEqual(events[2]["start_sec_30fps"], 0)
        self.assertIn("not inferred", events[0]["label_end_frame_semantics"])

    def test_coarse_labels_parse_tsv_and_keep_multword_action(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / f"assembly_{SESSION}.txt"
            path.write_text("000003000\t000003150\tattach bulldozer track\n", encoding="utf-8")
            events = parse_coarse_labels([path], SESSION)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["action_cls"], "attach bulldozer track")
        self.assertEqual(events[0]["start_sec_30fps"], 100)

    def test_build_manifest_joins_windows_and_prepares_unchecked_review_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fine = root / "train.csv"
            self.make_fine(fine)
            coarse = root / f"disassembly_{SESSION}.txt"
            coarse.write_text("4200 4300 remove screw from component\n", encoding="utf-8")
            windows = root / "windows.json"
            windows.write_text(json.dumps({"W03": {"start_sec": 100, "end_sec": 105},
                                           "W04": {"start_sec": 140, "end_sec": 145}}), encoding="utf-8")
            out = root / "prepared"
            result = build_manifest([fine], [coarse], SESSION, windows, out)
            self.assertEqual(len(result["events"]), 4)
            self.assertEqual({row["window_id"] for row in result["event_window_candidates"]}, {"W03", "W04"})
            self.assertEqual(len(result["event_window_candidates"]), 3)
            with (out / "evidence_review_template.csv").open(encoding="utf-8-sig", newline="") as stream:
                review_rows = list(csv.DictReader(stream))
            self.assertEqual(len(review_rows), 18)
            self.assertTrue(all(row["visibility"] == "unchecked" for row in review_rows))
            self.assertIn("source_pts_and_clip_offset_unverified",
                          result["event_window_candidates"][0]["mapping_status"])
            self.assertEqual(len(result["source_files"][0]["sha256"]), 64)
            self.assertTrue((out / "annotation_manifest.json").is_file())
            self.assertTrue((out / "event_window_candidates.csv").is_file())
            self.assertTrue((out / "evidence_review_template.csv").is_file())
            with self.assertRaises(FileExistsError):
                build_manifest([fine], [coarse], SESSION, windows, out)

    def test_rejects_wrong_coarse_session_and_bad_fine_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            coarse = root / "assembly_wrong-session.txt"
            coarse.write_text("0 10 action\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_coarse_labels([coarse], SESSION)
            fine = root / "bad.csv"
            fine.write_text("video,start_frame\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_fine_csv([fine], SESSION)


if __name__ == "__main__":
    unittest.main()
