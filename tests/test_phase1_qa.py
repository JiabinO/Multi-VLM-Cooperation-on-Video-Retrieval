import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import phase1_qa as m

CONFIG = Path(__file__).resolve().parents[1] / "configs/phase1_qa_v1.json"


class Runtime:
    metadata = {"backend": "offline_test_double"}
    loads = 0

    def __init__(self, model):
        Runtime.loads += 1

    def __call__(self, records, tokens, out, prompt):
        assert "reference_answer" not in prompt
        if out.name == "C10115":
            raise RuntimeError("injected generation failure")
        return "OFFLINE RESPONSE", {"input_tokens": len(records), "generate_seconds": 0}


class BatchQATests(unittest.TestCase):
    def args(self, temp, extract_only=False):
        clips = Path(temp) / "clips"
        clips.mkdir()
        for v in m.VIEWS:
            (clips / f"W06_{v}.mp4").touch()
        return argparse.Namespace(config=CONFIG, clips=clips, question_ids=["W06-Q1", "W06-Q2"],
            frames=8, max_new_tokens=128, model=Path("fake"), output=Path(temp) / "out", extract_only=extract_only)

    def test_sampling_reused_model_once_and_failures_continue(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(temp)
            calls = []
            def extract(video, dest, count, clip_start):
                calls.append((video.name, clip_start))
                dest.mkdir(parents=True)
                return {"frames": [{"frame_id": i, "file": f"frames/{i}.png", "nominal_time_sec": clip_start+i/2} for i in range(count)]}
            Runtime.loads = 0
            self.assertEqual(m.run(args, extract, Runtime), 1)
            self.assertEqual(Runtime.loads, 1)
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(t == 220 for _, t in calls))
            rows = json.loads((args.output / "review.json").read_text(encoding="utf-8"))["rows"]
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(r["verdict"] == "pending" for r in rows))
            self.assertEqual(sum(r["status"] == "failed" for r in rows), 2)
            self.assertEqual([r["metrics"]["input_tokens"] for r in rows[:5] if r["status"] == "completed"], [24, 8, 8, 16])
            self.assertEqual(rows[-1]["raw_response"], "OFFLINE RESPONSE")
            with self.assertRaises(FileExistsError):
                m.run(args, extract, Runtime)

    def test_extract_only_and_missing_input(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(temp, True)
            (args.clips / "W06_C10118.mp4").unlink()
            def extractor(video, dest, count, clip_start):
                dest.mkdir(parents=True)
                return {"frames": [{"frame_id": 0, "file": "x.png", "nominal_time_sec": 220}]}
            def forbidden(_):
                self.fail("model loaded")
            self.assertEqual(m.run(args, extractor, forbidden), 1)
            self.assertFalse(list(args.output.glob("*/*/raw_response.txt")))

    def test_config_rejects_unknown_selection_or_offset(self):
        with self.assertRaises(ValueError):
            m.load_config(CONFIG, ["W99-Q1"])
        with tempfile.TemporaryDirectory() as temp:
            data = json.loads(CONFIG.read_text(encoding="utf-8"))
            data["questions"][0]["source_start_sec"] = 0
            path = Path(temp) / "bad.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                m.load_config(path)

    def test_prompts_do_not_include_reference_or_scoring_notes(self):
        _, qs = m.load_config(CONFIG)
        self.assertEqual(len(qs), 6)
        for q in qs:
            prompt = m.prompt_for(q)
            self.assertIn(q["question"], prompt)
            self.assertNotIn(q["reference_answer"], prompt)
            self.assertNotIn(q["scoring_notes"], prompt)

    def test_all_six_questions_complete_thirty_tasks(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(temp)
            args.question_ids = None
            for window in ("W03", "W04", "W05", "W06"):
                for v in m.VIEWS:
                    (args.clips / f"{window}_{v}.mp4").touch()
            calls = []
            def extractor(video, dest, count, clip_start):
                calls.append(video.name)
                dest.mkdir(parents=True)
                return {"frames": [{"frame_id": 0, "file": "x.png", "nominal_time_sec": clip_start}]}
            class Success(Runtime):
                def __call__(self, records, tokens, out, prompt):
                    return "OFFLINE", {"input_tokens": len(records)}
            self.assertEqual(m.run(args, extractor, Success), 0)
            rows = json.loads((args.output / "review.json").read_text(encoding="utf-8"))["rows"]
            self.assertEqual(len(calls), 12)
            self.assertEqual(len(rows), 30)
            self.assertTrue(all(r["status"] == "completed" and r["verdict"] == "pending" for r in rows))


if __name__ == "__main__":
    unittest.main()
