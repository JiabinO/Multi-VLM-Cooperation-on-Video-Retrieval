"""Offline orchestration contracts; no model, CUDA, video data or installs."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import w02_qa as m


class FakeRuntime:
    metadata = {"backend": "offline-test-double-NOT-GPU"}

    def __init__(self, path):
        pass

    def __call__(self, records, max_tokens, out):
        if out.name == "C10115":
            raise RuntimeError("injected generation failure")
        return "OFFLINE TEST DOUBLE", {"input_tokens": len(records), "generate_seconds": 0}


def fake_extract(video, out, count):
    return {"frames": [{"frame_id": i, "file": f"frames/{i}.png",
                        "nominal_time_sec": 60 + i * 4 / count} for i in range(count)]}


class W02Contracts(unittest.TestCase):
    def args(self, out, extract_only=False):
        return argparse.Namespace(output=out, frames=8, max_new_tokens=128,
                                  model=Path("offline"), extract_only=extract_only,
                                  **{v: Path(v + ".mp4") for v in m.VIEWS})

    def test_groups_failure_continuation_and_raw_output(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            self.assertEqual(m.run(self.args(out), fake_extract, FakeRuntime), 1)
            run = json.loads((out / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(len(run["groups"]), 5)
            self.assertEqual(run["runtime"]["backend"], "offline-test-double-NOT-GPU")
            self.assertEqual([len(json.loads((out / g / "frames.json").read_text(encoding="utf-8")))
                              for g in m.GROUPS], [24, 8, 8, 8, 16])
            self.assertIn("injected", (out / "C10115/error.log").read_text(encoding="utf-8"))
            self.assertEqual((out / "C10095+C10115/raw_response.txt").read_text(encoding="utf-8"), "OFFLINE TEST DOUBLE")
            self.assertNotIn(m.REFERENCE, m.PROMPT)
            with self.assertRaises(FileExistsError):
                m.run(self.args(out), fake_extract, FakeRuntime)

    def test_extract_only_never_loads_model(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            def forbidden(_):
                self.fail("extract-only loaded model")
            self.assertEqual(m.run(self.args(out, True), fake_extract, forbidden), 0)
            self.assertFalse(list(out.glob("*/raw_response.txt")))
            self.assertEqual(json.loads((out / "full/run.json").read_text(encoding="utf-8"))["status"], "extracted")

    def test_extraction_and_load_failure_are_logged(self):
        for fail_extract in (True, False):
            with self.subTest(fail_extract=fail_extract), tempfile.TemporaryDirectory() as temp:
                out = Path(temp) / "run"
                def broken(*args):
                    raise RuntimeError("offline injected failure")
                self.assertEqual(m.run(self.args(out), broken if fail_extract else fake_extract, broken), 1)
                for name in m.GROUPS:
                    self.assertIn("offline injected failure", (out / name / "error.log").read_text(encoding="utf-8"))
                    self.assertIsNone(json.loads((out / name / "metrics.json").read_text(encoding="utf-8"))["input_tokens"])

    def test_interval_contract_from_smoke(self):
        from video_smoke import frame_indices
        ids = frame_indices(30, 2100, 60, 65, 8)
        self.assertEqual((ids[0], ids[-1]), (1800, 1949))
        self.assertEqual(len(set(ids)), 8)

    def test_w02_clips_use_explicit_source_offset(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            args = self.args(out, True)
            args.w02_clips = True
            calls = []
            def clipped(video, dest, count, clip_start):
                calls.append(clip_start)
                return fake_extract(video, dest, count)
            self.assertEqual(m.run(args, clipped, FakeRuntime), 0)
            self.assertEqual(calls, [60, 60, 60])


if __name__ == "__main__":
    unittest.main()

