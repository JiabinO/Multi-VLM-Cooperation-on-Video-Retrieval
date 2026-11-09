import unittest

from scripts.select_video_evidence import build_plans, select
from scripts.analyze_video_qa import summarize


def sample():
    return [{"query_id": "q1", "items": [
        {"item_id": "a0", "view_id": "a", "start_s": 0, "end_s": 1, "bytes": 100, "relevance_score": 0.2},
        {"item_id": "a1", "view_id": "a", "start_s": 1, "end_s": 2, "bytes": 100, "relevance_score": 0.1},
        {"item_id": "b0", "view_id": "b", "start_s": 0, "end_s": 1, "bytes": 100, "relevance_score": 0.9},
        {"item_id": "b1", "view_id": "b", "start_s": 1, "end_s": 2, "bytes": 100, "relevance_score": 0.2},
    ]}]


class VideoEvidenceSelectionTests(unittest.TestCase):
    def test_budget_never_exceeded_and_full_is_full(self):
        plans = build_plans(sample(), [0.25, 0.5, 1.0])
        for plan in plans:
            if plan["policy"] == "full_input":
                self.assertEqual(plan["uplink_bytes"], plan["full_input_bytes"])
            else:
                self.assertLessEqual(plan["uplink_bytes"], plan["budget_bytes"])

    def test_tiny_budget_is_explicitly_infeasible(self):
        plan = next(p for p in build_plans(sample(), [0.01])
                    if p["policy"] == "query_driven_selection")
        self.assertFalse(plan["feasible"])
        self.assertEqual(plan["selected_item_ids"], [])

    def test_query_score_prefers_value_per_byte(self):
        ids = select(sample()[0]["items"], "query_driven_selection", 100)
        self.assertEqual(ids, ["b0"])

    def test_uniform_sampling_covers_views_first(self):
        ids = select(sample()[0]["items"], "uniform_temporal_sampling", 100)
        self.assertEqual(set(ids), {"a0"})  # deterministic tie order by view

    def test_paired_analysis_checks_answers_and_budget(self):
        plans = build_plans(sample(), [0.5])
        predictions = [{**plan, "answer": "A", "correct_answer": "a"} for plan in plans]
        report = summarize(plans, predictions)
        self.assertEqual(report["n_queries"], 1)
        self.assertTrue(all(row["accuracy"] == 1 for row in report["conditions"]))

    def test_infeasible_budget_is_counted_without_fabricated_accuracy(self):
        plans = [p for p in build_plans(sample(), [0.01])
                 if p["policy"] in ("full_input", "query_driven_selection")]
        predictions = [{**p, "answer": "A" if p["feasible"] else None,
                        "correct_answer": "A", "dataset": "egoexo"} for p in plans]
        report = summarize(plans, predictions)
        query = next(c for c in report["conditions"] if c["policy"] == "query_driven_selection")
        self.assertEqual(query["n_evaluated"], 0)
        self.assertIsNone(query["accuracy"])


if __name__ == "__main__":
    unittest.main()
