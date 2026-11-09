"""Offline contracts for paired view-profile prompt experiment."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import phase1_view_prompt_ab as m


def extractor(video, out, count, clip_start):
    out.mkdir(parents=True)
    return {"frames": [{"frame_id": i, "frame_index": i, "file": f"frames/{i:03d}.png",
                        "nominal_time_sec": i / 2, "sha256": f"hash-{i}"} for i in range(count)]}


class Runtime:
    metadata = {"backend": "offline_test_double"}
    loads = 0
    calls = []

    def __init__(self, model):
        Runtime.loads += 1

    def __call__(self, records, tokens, out, prompt):
        Runtime.calls.append((out.parent.name, out.name, [r["sha256"] for r in records], prompt))
        raw = {"visible_facts": [{"category": "object", "claim": "red object visible",
                "frame_ids": [0], "evidence": "frame 0 shows a red object", "confidence": 0.7}],
               "uncertain_items": [{"detail": "whether it was picked up", "frame_ids": [1],
                                    "reason": "no continuous motion sampled"}]}
        return json.dumps(raw, ensure_ascii=False), {"input_tokens": 20, "generate_seconds": 0.1}


class ViewPromptABTests(unittest.TestCase):
    def setup_args(self, temp, extract_only=False):
        root = Path(temp)
        clips = root / "clips"
        clips.mkdir()
        for view in m.VIEWS:
            (clips / f"W03_{view}.mp4").touch()
        profiles = root / "profiles.json"
        profiles.write_text(json.dumps({"profiles": {v: {"approved": True,
            "view_coverage": f"approved field of view for {v}"} for v in m.VIEWS}}, ensure_ascii=False), encoding="utf-8")
        return argparse.Namespace(clips=clips, profiles=profiles, windows=["W03"], views=list(m.VIEWS),
            model=Path("fake"), frames=2, max_new_tokens=128, output=root / "out", extract_only=extract_only)

    def test_paired_conditions_share_identical_frames_and_one_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.setup_args(temp)
            Runtime.loads, Runtime.calls = 0, []
            self.assertEqual(m.run(args, extractor, Runtime), 0)
            self.assertEqual(Runtime.loads, 1)
            self.assertEqual(len(Runtime.calls), 6)
            for view in m.VIEWS:
                pair_calls = [c for c in Runtime.calls if c[0] == f"W03_{view}"]
                self.assertEqual({c[1] for c in pair_calls}, {"no_profile", "approved_profile"})
                self.assertEqual(pair_calls[0][2], pair_calls[1][2])
                baseline = next(c for c in pair_calls if c[1] == "no_profile")[3]
                with_profile = next(c for c in pair_calls if c[1] == "approved_profile")[3]
                self.assertNotIn("approved field of view", baseline)
                self.assertIn("approved field of view", with_profile)
            review = json.loads((args.output / "review.json").read_text(encoding="utf-8"))
            self.assertFalse(review["is_gold"])
            self.assertEqual(len(review["pairs"]), 3)
            self.assertEqual(review["pairs"][0]["status"], "completed")
            self.assertTrue((args.output / "W03_C10095/approved_profile/raw_response.txt").is_file())

    def test_pending_profile_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.setup_args(temp)
            data = json.loads(args.profiles.read_text(encoding="utf-8"))
            data["profiles"]["C10095"]["approved"] = False
            args.profiles.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "approved=true"):
                m.run(args, extractor, Runtime)
            self.assertFalse(args.output.exists())

    def test_parser_checks_uncertainty_and_frame_references(self):
        records = [{"frame_id": 0}, {"frame_id": 1}]
        good = {"visible_facts": [{"category": "action", "claim": "x", "frame_ids": [0],
                "evidence": "visible", "confidence": 0.6}], "uncertain_items": []}
        self.assertEqual(m.parse_observation(json.dumps(good), records), good)
        bad = dict(good, uncertain_items=[{"detail": "x", "frame_ids": [8], "reason": "blur"}])
        with self.assertRaises(ValueError):
            m.parse_observation(json.dumps(bad), records)

    def test_extract_only_needs_no_profile_or_model(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.setup_args(temp, True)
            args.profiles = None
            def forbidden(_):
                self.fail("extract-only must not load the model")
            self.assertEqual(m.run(args, extractor, forbidden), 0)
            self.assertEqual(json.loads((args.output / "run.json").read_text()) ["status"], "extracted")


if __name__ == "__main__":
    unittest.main()
