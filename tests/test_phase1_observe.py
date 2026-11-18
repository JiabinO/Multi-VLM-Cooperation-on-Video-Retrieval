"""Offline review workflow tests; no CUDA or video packages required."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import phase1_observe as m


def extractor(video, out, count, clip_start):
    out.mkdir(parents=True)
    return {"frames": [{"frame_id": 0, "file": "frames/000.png", "nominal_time_sec": clip_start}]}


class Runtime:
    metadata = {"backend": "offline_test_double"}
    loads = 0

    def __init__(self, model):
        Runtime.loads += 1

    def __call__(self, records, tokens, out, prompt):
        assert prompt == m.PROMPT
        if "C10115" in out.name:
            return "invalid JSON", {"input_tokens": 1}
        return json.dumps({"observations": [{"category": "action", "claim": "draft only",
                            "frame_ids": [0], "evidence": "第0帧可见手接触组件",
                            "confidence": 0.7, "uncertainty": "unclear"}]}), {"input_tokens": 1}


class ObservationTests(unittest.TestCase):
    def setup_args(self, temp, extract_only=False):
        clips = Path(temp) / "clips"
        clips.mkdir()
        for v in m.VIEWS:
            (clips / f"W03_{v}.mp4").touch()
        return argparse.Namespace(clips=clips, output=Path(temp) / "out", model=Path("fake"),
                                  frames=8, max_new_tokens=768, windows=["W03"], extract_only=extract_only)

    def test_invalid_draft_keeps_raw_and_later_views_continue(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.setup_args(temp)
            Runtime.loads = 0
            self.assertEqual(m.run(args, extractor, Runtime), 1)
            self.assertEqual(Runtime.loads, 1)
            review = json.loads((args.output / "review.json").read_text(encoding="utf-8"))
            self.assertFalse(review["is_gold"])
            self.assertEqual([t["status"] for t in review["tasks"]], ["drafted", "parse_failed", "drafted"])
            self.assertEqual(review["tasks"][0]["observations"][0]["decision"], "pending")
            self.assertEqual(review["tasks"][0]["source_offset_sec"], 100)
            self.assertEqual((args.output / "W03_C10115/raw_response.txt").read_text(), "invalid JSON")
            self.assertIn("模型遗漏", (args.output / "review.md").read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                m.run(args, extractor, Runtime)

    def test_missing_video_is_logged_without_loading_in_extract_only(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.setup_args(temp, True)
            (args.clips / "W03_C10095.mp4").unlink()
            def forbidden(_):
                self.fail("extract-only loaded model")
            self.assertEqual(m.run(args, extractor, forbidden), 1)
            self.assertIn("FileNotFoundError", (args.output / "W03_C10095/error.log").read_text())
            self.assertTrue((args.output / "W03_C10118/task.json").is_file())

    def test_load_failure_is_preserved_for_all_tasks(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.setup_args(temp)
            def broken(_):
                raise RuntimeError("offline load failure")
            self.assertEqual(m.run(args, extractor, broken), 1)
            for v in m.VIEWS:
                self.assertIn("offline load failure", (args.output / f"W03_{v}/error.log").read_text())

    def test_schema_and_frame_references(self):
        records = [{"frame_id": 0}]
        good = {"category": "object", "claim": "x", "frame_ids": [0],
                "evidence": "第0帧可见该物体", "confidence": 0.8, "uncertainty": ""}
        self.assertEqual(m.parse_draft('```json\n{"observations": []}\n```', records), {"observations": []})
        for patch in ({"frame_ids": [9]}, {"frame_ids": [True]}, {"claim": ""},
                      {"category": "invented"}, {"uncertainty": None},
                      {"evidence": ""}, {"confidence": -0.1}, {"confidence": 1.1},
                      {"confidence": True}, {"confidence": "0.8"}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                m.parse_draft(json.dumps({"observations": [dict(good, **patch)]}), records)

    def test_review_form_exposes_confidence_evidence_and_caveat(self):
        task = {"task_id": "W03_C10095", "status": "drafted", "observations": [{
            "id": "W03_C10095-000", "category": "action", "claim": "draft",
            "frame_ids": [0], "evidence": "手接触螺丝", "confidence": 0.82,
            "uncertainty": "拆下状态不清", "decision": "pending"}]}
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            m.write_review(out, [task])
            md = (out / "review.md").read_text(encoding="utf-8")
            data = json.loads((out / "review.json").read_text(encoding="utf-8"))
            self.assertIn("模型置信度（自评，非准确率）：0.82", md)
            self.assertIn("模型描述的视觉证据：手接触螺丝", md)
            self.assertIn("不代表校准准确率", data["instructions"])


if __name__ == "__main__":
    unittest.main()
