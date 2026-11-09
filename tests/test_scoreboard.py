import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from scoreboard_eval import evaluate, parse_readings
from scoreboard_probe import load_source


class ScoreboardTests(unittest.TestCase):
    def reference(self):
        return json.loads((ROOT / "data/manifests/video_mme_scoreboard_review.json").read_text(encoding="utf-8"))

    def predictions(self):
        expected = {f["frame_id"]: f["expected"] for f in self.reference()["frames"] if "expected" in f}
        return {"frames": [{"frame_id": i, "status": "not_visible", "eng_score": None,
                            "den_score": None, "match_clock": None, "uncertainty": "",
                            **expected.get(i, {})} for i in range(8)]}

    def test_only_confirmed_scoreboard_labels_count(self):
        data = self.predictions()
        data["frames"][1]["match_clock"] = "05:39"
        parsed = parse_readings(json.dumps(data), list(range(8)))
        report = evaluate(parsed, self.reference())
        self.assertEqual(report["scored_frames"], 3)
        self.assertEqual(report["exact_match_frames"], 3)
        self.assertEqual(report["unscored_frame_ids"], [0, 4, 5, 6, 7])

    def test_wrong_clock_is_scored_as_wrong(self):
        data = self.predictions()
        data["frames"][2]["match_clock"] = "5:44"
        self.assertEqual(evaluate(data, self.reference())["exact_match_frames"], 2)

    def test_schema_rejects_missing_duplicate_unknown_and_invalid_clock(self):
        variants = []
        x = self.predictions(); x["frames"].pop(); variants.append(x)
        x = self.predictions(); x["frames"][0]["frame_id"] = 1; variants.append(x)
        x = self.predictions(); x["frames"][0]["frame_id"] = 9; variants.append(x)
        x = self.predictions(); x["frames"][1]["match_clock"] = "5:99"; variants.append(x)
        x = self.predictions(); x["frames"][1]["eng_score"] = True; variants.append(x)
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                parse_readings(json.dumps(variant), list(range(8)))

    def test_source_binding_checksums_and_cli_without_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            frames = []
            for ref in self.reference()["frames"]:
                file = f"{ref['frame_id']}.png"
                (source / file).write_bytes(b"fixture-only-checksum")
                frames.append({"frame_id": ref["frame_id"], "frame_index": ref["frame_index"],
                               "file": file, "sha256": hashlib.sha256(b"fixture-only-checksum").hexdigest()})
            (source / "frames.json").write_text(json.dumps({"frames": frames}), encoding="utf-8")
            (source / "run.json").write_text(json.dumps({"config": {"video": "/data/_8lBR0E_Tx8.mp4"}}), encoding="utf-8")
            out = root / "validation"
            command = [sys.executable, str(ROOT / "scripts/scoreboard_probe.py"), "--source", str(source),
                       "--output", str(out), "--validate-only"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((out / "run.json").read_text(encoding="utf-8"))["status"], "validated")
            # Existing output must not be overwritten.
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            (source / "0.png").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                load_source(source, self.reference())


if __name__ == "__main__":
    unittest.main()
