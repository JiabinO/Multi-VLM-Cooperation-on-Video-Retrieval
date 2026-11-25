"""Offline tests for camera-view profile drafts; no CUDA or video packages required."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import phase1_view_profile as m


def extractor(video, out, count, clip_start):
    out.mkdir(parents=True)
    return {"frames": [{"frame_id": i, "file": f"frames/{i:03d}.png",
                        "nominal_time_sec": clip_start + i * 0.5} for i in range(count)]}


class Runtime:
    metadata = {"backend": "offline_test_double"}
    loads = 0
    calls = []

    def __init__(self, model):
        Runtime.loads += 1

    def __call__(self, records, tokens, out, prompt):
        Runtime.calls.append((out.name, len(records), prompt))
        return json.dumps({"observations": [{"category": "scene_layout", "claim": "stable draft",
            "frame_ids": [0], "evidence": "input frame 0 shows the table", "confidence": 0.8,
            "uncertainty": "camera position not certain"}]}), {"input_tokens": 12, "generate_seconds": 0.1}


class ViewProfileTests(unittest.TestCase):
    def setup_args(self, temp, extract_only=False):
        clips = Path(temp) / "clips"
        clips.mkdir()
        for window in ("W03", "W04"):
            for view in m.VIEWS:
                (clips / f"{window}_{view}.mp4").touch()
        return argparse.Namespace(clips=clips, windows=["W03", "W04"], frames_per_window=2,
            max_new_tokens=100, output=Path(temp) / "out", model=Path("fake"), extract_only=extract_only)

    def test_profiles_combine_windows_and_reuse_one_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.setup_args(temp)
            Runtime.loads, Runtime.calls = 0, []
            self.assertEqual(m.run(args, extractor, Runtime), 0)
            self.assertEqual(Runtime.loads, 1)
            self.assertEqual([c[1] for c in Runtime.calls], [4, 4, 4])
            self.assertTrue(all(m.PROMPT in c[2] for c in Runtime.calls))
            review = json.loads((args.output / "review.json").read_text(encoding="utf-8"))
            self.assertFalse(review["is_gold"])
            self.assertEqual([t["status"] for t in review["tasks"]], ["drafted"] * 3)
            frames = json.loads((args.output / "C10095/frames.json").read_text(encoding="utf-8"))
            self.assertEqual([f["window"] for f in frames], ["W03", "W03", "W04", "W04"])
            md = (args.output / "review.md").read_text(encoding="utf-8")
            self.assertIn("不能据此认定信息在其他机位不可见", md)
            self.assertIn("模型置信度（自评，非准确率）", md)
            with self.assertRaises(FileExistsError):
                m.run(args, extractor, Runtime)

    def test_invalid_frame_reference_is_rejected(self):
        records = [{"profile_frame_id": 0}]
        good = {"category": "scene_layout", "claim": "x", "frame_ids": [0],
                "evidence": "visible in frame", "confidence": 0.5, "uncertainty": ""}
        self.assertEqual(m.parse_profile('{"observations": []}', records), {"observations": []})
        for patch in ({"frame_ids": [1]}, {"frame_ids": [True]}, {"category": "unique_view"},
                      {"confidence": True}, {"evidence": ""}, {"uncertainty": None}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                m.parse_profile(json.dumps({"observations": [dict(good, **patch)]}), records)

    def test_extract_only_does_not_load_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.setup_args(temp, True)
            def forbidden(_):
                self.fail("extract-only loaded model")
            self.assertEqual(m.run(args, extractor, forbidden), 0)
            self.assertEqual(json.loads((args.output / "run.json").read_text()) ["status"], "extracted")
            self.assertTrue((args.output / "C10118/frames.json").is_file())


if __name__ == "__main__":
    unittest.main()
