from pathlib import Path
import tempfile
import unittest

from scripts.video_evidence import (parse_answer, parse_relevance, qa_prompt,
                                safe_video_path, selected_views, split_for_sample)


class VideoEvidenceHelpersTests(unittest.TestCase):
    def test_relevance_output_requires_bounded_numeric_score(self):
        self.assertEqual(parse_relevance('{"score":0.7,"reason":"visible evidence"}')[0], 0.7)
        for raw in ('{"score":1.2,"reason":"x"}', '{"score":"high","reason":"x"}', 'not json'):
            with self.assertRaises((ValueError, TypeError)):
                parse_relevance(raw)

    def test_answer_output_is_exactly_one_option(self):
        self.assertEqual(parse_answer('{"answer":"C","confidence":"medium"}'), ("C", "medium"))
        with self.assertRaises(ValueError):
            parse_answer('{"answer":"AB","confidence":"high"}')

    def test_answer_key_does_not_enter_qa_prompt(self):
        prompt = qa_prompt("What happened?", {"A": "one", "B": "two", "C": "three", "D": "four"})
        self.assertIn("What happened?", prompt)
        self.assertNotIn("correct_answer", prompt)

    def test_designated_views_must_be_mapped(self):
        self.assertEqual([x["view"] for x in selected_views({"views": [2, 0]}, [
            {"view": "0", "target": "a.mp4"}, {"view": "2", "target": "b.mp4"}])], ["0", "2"])
        with self.assertRaises(ValueError):
            selected_views({"views": [1]}, [{"view": "0", "target": "a.mp4"}])

    def test_scene_split_is_stable_and_paths_cannot_escape_root(self):
        self.assertEqual(split_for_sample("scene-1", 7, .6, .2),
                         split_for_sample("scene-1", 7, .6, .2))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(safe_video_path(root, "nested/video.mp4"), (root / "nested" / "video.mp4").resolve())
            with self.assertRaises(ValueError):
                safe_video_path(root, "../outside.mp4")


if __name__ == "__main__":
    unittest.main()
