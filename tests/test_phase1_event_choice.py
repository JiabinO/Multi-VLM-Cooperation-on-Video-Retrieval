import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import phase1_event_choice as experiment


class EventChoiceTests(unittest.TestCase):
    def test_catalog_unions_camera_specific_fine_labels_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"events": [
                {"granularity": "fine", "view_id": "C10095", "action_cls": "pick up screw",
                 "start_frame_30fps": 30, "end_frame_30fps": 45},
                {"granularity": "fine", "view_id": "C10115", "action_cls": "pick up screw",
                 "start_frame_30fps": 32, "end_frame_30fps": 47},
                {"granularity": "fine", "view_id": "C10118", "action_cls": "position screw",
                 "start_frame_30fps": 50, "end_frame_30fps": 75},
                {"granularity": "coarse", "view_id": "all_views_coarse", "action_cls": "assembly",
                 "start_frame_30fps": 0, "end_frame_30fps": 100},
            ]}), encoding="utf-8")
            _, per_view, labels = experiment.load_catalog(path)
        self.assertEqual(labels, ["pick up screw", "position screw"])
        self.assertIn("pick up screw", per_view["C10095"])
        self.assertNotIn("position screw", per_view["C10095"])

    def test_option_order_is_deterministic_and_shared(self):
        first = experiment.build_options(["attach part", "pick up screw"], seed=21)
        second = experiment.build_options(["attach part", "pick up screw"], seed=21)
        self.assertEqual(first, second)
        self.assertEqual({item["action_cls"] for item in first}, {"attach part", "pick up screw"})

    def test_sampling_coverage_distinguishes_missing_one_and_multiple_samples(self):
        events = {
            "brief": [{"start_frame": 30, "end_frame": 38}],
            "long": [{"start_frame": 30, "end_frame": 120}],
            "missed": [{"start_frame": 31, "end_frame": 38}],
        }
        coverage = experiment.sample_coverage(events, [0.0, 1.0, 2.0, 4.0])
        self.assertEqual(coverage["brief"]["coverage_status"], "one_sampled_frame_only")
        self.assertEqual(coverage["long"]["coverage_status"], "at_least_two_sampled_frames")
        self.assertEqual(coverage["missed"]["coverage_status"], "no_sampled_frame_not_evaluable")

    def test_choice_parser_requires_all_unique_options_and_valid_evidence(self):
        options = [{"option_id": "E001", "action_cls": "pick up screw"},
                   {"option_id": "E002", "action_cls": "position screw"}]
        records = [{"frame_id": 0}, {"frame_id": 1}]
        raw = json.dumps({"choices": [
            {"option_id": "E001", "decision": "visible", "confidence": 0.8,
             "evidence_frame_ids": [1], "evidence": "手中拿着螺丝"},
            {"option_id": "E002", "decision": "not_observed", "confidence": 0.5,
             "evidence_frame_ids": [], "evidence": "采样帧中没有直接证据"},
        ]})
        self.assertEqual(len(experiment.parse_choices(raw, options, records)["choices"]), 2)
        with self.assertRaises(ValueError):
            experiment.parse_choices(raw.replace('"E002"', '"E001"'), options, records)
        with self.assertRaises(ValueError):
            experiment.parse_choices(raw.replace('"evidence_frame_ids": [1]', '"evidence_frame_ids": [9]'), options, records)


if __name__ == "__main__":
    unittest.main()
