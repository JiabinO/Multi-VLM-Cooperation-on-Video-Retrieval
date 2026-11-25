import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("video_smoke", Path(__file__).resolve().parents[1] / "scripts/video_smoke.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SmokeContracts(unittest.TestCase):
    def test_sampling_boundaries_and_distinctness(self):
        ids = m.frame_indices(29.97002997, 17618, 0, 30, 8)
        self.assertEqual(len(set(ids)), 8)
        self.assertTrue(all(0 <= i / 29.97002997 < 30 for i in ids))
        for args in [(30, 900, -1, 10, 8), (30, 900, 0, 31, 8), (30, 900, 0, .01, 8)]:
            with self.assertRaises(ValueError):
                m.frame_indices(*args)

    def test_invalid_reference_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            m.parse_evidence('{"observations":[{"claim":"x","frame_ids":[8],"uncertainty":""}]}',
                             [{"frame_id": 0, "nominal_time_sec": 0}])

    def test_evidence_bounds_come_from_sampled_frames(self):
        result = m.parse_evidence('{"observations":[{"claim":"x","frame_ids":[1,0],"uncertainty":""}]}',
                                 [{"frame_id": 0, "nominal_time_sec": 2}, {"frame_id": 1, "nominal_time_sec": 6}])
        self.assertEqual(result["observations"][0]["sampled_time_bounds_sec"], [2, 6])


if __name__ == "__main__":
    unittest.main()
